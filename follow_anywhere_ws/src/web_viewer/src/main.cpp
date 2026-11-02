// web_viewer: serves www/ over HTTP and streams topics to the browser over a
// WebSocket on the same port. Subscriptions are made on demand, when the
// first browser asks for a topic, and dropped when the last one leaves.
//
// Browser -> server (text JSON):
//   {"op":"sub","topic":"/x"}   {"op":"unsub","topic":"/x"}
// Server -> browser:
//   text   {"op":"hello","defaults":[...]}
//   text   {"op":"topics","topics":[["/x","pkg/msg/T"],...]}
//   text   {"op":"subscribed","topic":"/x","type":"...","kind":"image|cloud|json"}
//   text   {"op":"msg","topic":"/x","data":{...}}           (kind json)
//   text   {"op":"error","topic":"/x","error":"..."}
//   binary [u8 kind][u8 0][u16le topic_len][topic][pad to 4][payload]
//          kind 1 = JPEG, 2 = PNG, 3 = float32 x,y,z,intensity points

#include <chrono>
#include <map>
#include <mutex>
#include <set>

#include <ament_index_cpp/get_package_share_path.hpp>
#include <rclcpp/rclcpp.hpp>

#include "encoders.hpp"
#include "msg_json.hpp"
#include "server.hpp"

using namespace std::chrono_literals;

namespace web_viewer
{

namespace
{

// Reads a string or number field from the flat JSON objects the browser sends.
std::string json_field(const std::string & js, const std::string & key)
{
  auto k = js.find("\"" + key + "\"");
  if (k == std::string::npos) {return {};}
  auto c = js.find(':', k);
  if (c == std::string::npos) {return {};}
  size_t i = js.find_first_not_of(" \t\r\n", c + 1);
  if (i == std::string::npos) {return {};}
  std::string v;
  if (js[i] == '"') {
    for (++i; i < js.size() && js[i] != '"'; ++i) {
      if (js[i] == '\\' && i + 1 < js.size()) {++i;}
      v += js[i];
    }
  } else {
    for (; i < js.size() && js[i] != ',' && js[i] != '}'; ++i) {v += js[i];}
  }
  return v;
}

Payload text(std::string s) {return std::make_shared<const std::string>(std::move(s));}

Payload binary_frame(uint8_t kind, const std::string & topic, const void * data, size_t size)
{
  size_t head = 4 + topic.size();
  size_t pad = (4 - head % 4) % 4;
  auto out = std::make_shared<std::string>();
  out->reserve(head + pad + size);
  out->push_back(static_cast<char>(kind));
  out->push_back(0);
  out->push_back(static_cast<char>(topic.size() & 0xff));
  out->push_back(static_cast<char>((topic.size() >> 8) & 0xff));
  out->append(topic);
  out->append(pad, '\0');
  out->append(static_cast<const char *>(data), size);
  return out;
}

enum class Kind { Image, Compressed, Cloud, Scan, Json };

Kind kind_of(const std::string & type)
{
  if (type == "sensor_msgs/msg/Image") {return Kind::Image;}
  if (type == "sensor_msgs/msg/CompressedImage") {return Kind::Compressed;}
  if (type == "sensor_msgs/msg/PointCloud2") {return Kind::Cloud;}
  if (type == "sensor_msgs/msg/LaserScan") {return Kind::Scan;}
  return Kind::Json;
}

const char * kind_name(Kind k)
{
  switch (k) {
    case Kind::Image: case Kind::Compressed: return "image";
    case Kind::Cloud: case Kind::Scan: return "cloud";
    default: return "json";
  }
}

}  // namespace

class WebViewer : public rclcpp::Node
{
public:
  WebViewer()
  : Node("web_viewer")
  {
    auto address = declare_parameter<std::string>("address", "0.0.0.0");
    auto port = declare_parameter<int>("port", 8080);
    auto www = declare_parameter<std::string>("www_root", "");
    image_hz_ = declare_parameter<double>("image_hz", 15.0);
    cloud_hz_ = declare_parameter<double>("cloud_hz", 10.0);
    json_hz_ = declare_parameter<double>("json_hz", 10.0);
    max_points_ = declare_parameter<int>("max_points", 40000);
    max_array_ = declare_parameter<int>("max_array", 64);
    img_opt_.jpeg_quality = declare_parameter<int>("jpeg_quality", 70);
    img_opt_.max_width = declare_parameter<int>("max_width", 960);
    img_opt_.depth_max_m = declare_parameter<double>("depth_max_m", 6.0);
    defaults_ = declare_parameter<std::vector<std::string>>(
      "default_topics",
      {"/camera/camera/color/image_raw", "/utlidar/cloud_base", "/sportmodestate"});

    if (www.empty()) {
      www = (ament_index_cpp::get_package_share_path("web_viewer") / "www").string();
    }

    ServerCallbacks cb;
    cb.on_open = [this](const SessionPtr & s) {on_open(s);};
    cb.on_text = [this](const SessionPtr & s, const std::string & m) {on_text(s, m);};
    cb.on_close = [this](const SessionPtr & s) {on_close(s);};
    server_ = std::make_unique<Server>(address, static_cast<uint16_t>(port), www, cb);
    server_->start();

    topics_timer_ = create_wall_timer(2s, [this] {refresh_topics(false);});
    RCLCPP_INFO(get_logger(), "web_viewer on http://%s:%ld (files from %s)",
      address.c_str(), port, www.c_str());
  }

  ~WebViewer() override
  {
    server_->stop();
  }

private:
  struct Stream
  {
    std::string topic, type;
    Kind kind;
    rclcpp::CallbackGroup::SharedPtr group;
    rclcpp::GenericSubscription::SharedPtr sub;
    std::unique_ptr<MsgToJson> json;
    std::map<uint64_t, std::weak_ptr<WsSession>> clients;
    std::chrono::steady_clock::time_point last{};
    bool warned = false;
  };

  // ------------------------------------------------------------ sessions

  void on_open(const SessionPtr & s)
  {
    std::string hello = "{\"op\":\"hello\",\"defaults\":[";
    for (size_t i = 0; i < defaults_.size(); ++i) {
      if (i) {hello += ',';}
      json_escape(hello, defaults_[i]);
    }
    hello += "]}";
    s->send(text(hello), false, false);
    {
      std::lock_guard<std::mutex> lk(mtx_);
      sessions_[s->id()] = s;
    }
    refresh_topics(true, s);
  }

  void on_close(const SessionPtr & s)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    sessions_.erase(s->id());
    for (auto it = streams_.begin(); it != streams_.end(); ) {
      it->second->clients.erase(s->id());
      it = it->second->clients.empty() ? streams_.erase(it) : std::next(it);
    }
  }

  void on_text(const SessionPtr & s, const std::string & msg)
  {
    auto op = json_field(msg, "op");
    auto topic = json_field(msg, "topic");
    if (topic.empty()) {return;}
    if (op == "sub") {
      subscribe(s, topic);
    } else if (op == "unsub") {
      std::lock_guard<std::mutex> lk(mtx_);
      auto it = streams_.find(topic);
      if (it != streams_.end()) {
        it->second->clients.erase(s->id());
        if (it->second->clients.empty()) {streams_.erase(it);}
      }
    }
  }

  void send_error(const SessionPtr & s, const std::string & topic, const std::string & err)
  {
    std::string m = "{\"op\":\"error\",\"topic\":";
    json_escape(m, topic);
    m += ",\"error\":";
    json_escape(m, err);
    m += '}';
    s->send(text(m), false, false);
  }

  // ------------------------------------------------------- subscriptions

  void subscribe(const SessionPtr & s, const std::string & topic)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = streams_.find(topic);
    if (it == streams_.end()) {
      auto types = get_topic_names_and_types();
      auto t = types.find(topic);
      if (t == types.end() || t->second.empty()) {
        send_error(s, topic, "topic not found");
        return;
      }
      auto st = std::make_shared<Stream>();
      st->topic = topic;
      st->type = t->second.front();
      st->kind = kind_of(st->type);
      try {
        if (st->kind == Kind::Json) {st->json = std::make_unique<MsgToJson>(st->type);}
        std::weak_ptr<Stream> weak = st;
        rclcpp::SubscriptionOptions opts;
        st->group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        opts.callback_group = st->group;
        st->sub = create_generic_subscription(topic, st->type, rclcpp::SensorDataQoS(),
          [this, weak](std::shared_ptr<const rclcpp::SerializedMessage> m) {
            if (auto sp = weak.lock()) {on_message(*sp, *m);}
          }, opts);
      } catch (const std::exception & e) {
        send_error(s, topic, std::string("cannot subscribe: ") + e.what());
        return;
      }
      RCLCPP_INFO(get_logger(), "streaming %s [%s]", topic.c_str(), st->type.c_str());
      it = streams_.emplace(topic, st).first;
    }
    it->second->clients[s->id()] = s;

    std::string m = "{\"op\":\"subscribed\",\"topic\":";
    json_escape(m, topic);
    m += ",\"type\":";
    json_escape(m, it->second->type);
    m += ",\"kind\":\"";
    m += kind_name(it->second->kind);
    m += "\"}";
    s->send(text(m), false, false);
  }

  void on_message(Stream & st, const rclcpp::SerializedMessage & raw)
  {
    double hz = st.kind == Kind::Json ? json_hz_ :
      (st.kind == Kind::Cloud || st.kind == Kind::Scan) ? cloud_hz_ : image_hz_;
    auto now = std::chrono::steady_clock::now();
    if (hz > 0 && now - st.last < std::chrono::duration<double>(1.0 / hz)) {return;}
    st.last = now;

    std::vector<SessionPtr> clients;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      for (auto & [id, w] : st.clients) {
        if (auto s = w.lock()) {clients.push_back(s);}
      }
    }
    if (clients.empty()) {return;}

    Payload out;
    bool is_binary = true;
    try {
      switch (st.kind) {
        case Kind::Image: {
            sensor_msgs::msg::Image img;
            image_ser_.deserialize_message(&raw, &img);
            auto jpg = encode_image(img, img_opt_);
            if (jpg.empty()) {
              warn_once(st, clients, "unsupported image encoding '" + img.encoding + "'");
              return;
            }
            out = binary_frame(1, st.topic, jpg.data(), jpg.size());
            break;
          }
        case Kind::Compressed: {
            sensor_msgs::msg::CompressedImage img;
            compressed_ser_.deserialize_message(&raw, &img);
            if (img.format.find("compressedDepth") != std::string::npos) {
              warn_once(st, clients, "compressedDepth can't be shown, use the raw depth topic");
              return;
            }
            uint8_t k = img.format.find("png") != std::string::npos ? 2 : 1;
            out = binary_frame(k, st.topic, img.data.data(), img.data.size());
            break;
          }
        case Kind::Cloud: {
            sensor_msgs::msg::PointCloud2 pc;
            cloud_ser_.deserialize_message(&raw, &pc);
            auto pts = cloud_points(pc, static_cast<size_t>(max_points_));
            if (pts.empty() && pc.width * pc.height > 0) {
              warn_once(st, clients, "cloud has no float32 x/y/z fields");
            }
            out = binary_frame(3, st.topic, pts.data(), pts.size() * sizeof(float));
            break;
          }
        case Kind::Scan: {
            sensor_msgs::msg::LaserScan scan;
            scan_ser_.deserialize_message(&raw, &scan);
            auto pts = scan_points(scan);
            out = binary_frame(3, st.topic, pts.data(), pts.size() * sizeof(float));
            break;
          }
        case Kind::Json: {
            std::string m = "{\"op\":\"msg\",\"topic\":";
            json_escape(m, st.topic);
            m += ",\"data\":";
            m += st.json->convert(raw, static_cast<size_t>(max_array_));
            m += '}';
            out = text(std::move(m));
            is_binary = false;
            break;
          }
      }
    } catch (const std::exception & e) {
      warn_once(st, clients, std::string("decode failed: ") + e.what());
      return;
    }
    for (auto & s : clients) {s->send(out, is_binary, true);}
  }

  void warn_once(Stream & st, const std::vector<SessionPtr> & clients, const std::string & err)
  {
    if (st.warned) {return;}
    st.warned = true;
    RCLCPP_WARN(get_logger(), "%s: %s", st.topic.c_str(), err.c_str());
    for (auto & s : clients) {send_error(s, st.topic, err);}
  }

  // --------------------------------------------------------- topic list

  void refresh_topics(bool force, const SessionPtr & only = nullptr)
  {
    std::string m = "{\"op\":\"topics\",\"topics\":[";
    bool first = true;
    for (auto & [name, types] : get_topic_names_and_types()) {
      if (name.find("/_") != std::string::npos) {continue;}  // hidden, e.g. /x/_buf_cpu
      if (!first) {m += ',';}
      first = false;
      m += '[';
      json_escape(m, name);
      m += ',';
      json_escape(m, types.empty() ? "" : types.front());
      m += ']';
    }
    m += "]}";

    std::vector<SessionPtr> targets;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!force && m == last_topics_) {return;}
      last_topics_ = m;
      if (only) {
        targets.push_back(only);
      } else {
        for (auto & [id, w] : sessions_) {
          if (auto s = w.lock()) {targets.push_back(s);}
        }
      }
    }
    auto p = text(std::move(m));
    for (auto & s : targets) {s->send(p, false, false);}
  }

  double image_hz_, cloud_hz_, json_hz_;
  int64_t max_points_, max_array_;
  ImageOptions img_opt_;
  std::vector<std::string> defaults_;

  rclcpp::Serialization<sensor_msgs::msg::Image> image_ser_;
  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> compressed_ser_;
  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> cloud_ser_;
  rclcpp::Serialization<sensor_msgs::msg::LaserScan> scan_ser_;

  std::mutex mtx_;
  std::map<std::string, std::shared_ptr<Stream>> streams_;
  std::map<uint64_t, std::weak_ptr<WsSession>> sessions_;
  std::string last_topics_;
  rclcpp::TimerBase::SharedPtr topics_timer_;
  std::unique_ptr<Server> server_;
};

}  // namespace web_viewer

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<web_viewer::WebViewer>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();
  exec.remove_node(node);
  node.reset();
  rclcpp::shutdown();
  return 0;
}
