// web_viewer: serves www/ over HTTP and streams topics to the browser over a
// WebSocket on the same port. Subscriptions are made on demand, when the
// first browser asks for a topic, and dropped when the last one leaves.
//
// Browser -> server (text JSON):
//   {"op":"sub","topic":"/x"}   {"op":"unsub","topic":"/x"}
//   optional in sub: "hz":n      max rate of this stream, 0 = every message
//                    "pair":"/y" (images) only send the image whose header
//                                stamp equals a message on /y, "" = unpair
//   {"op":"calibrate","track_id":7}   {"op":"clear_leader"}
//   {"op":"record_start"}   {"op":"record_stop","title":"first_follow"}
// Server -> browser:
//   text   {"op":"hello","defaults":[...]}
//   text   {"op":"topics","topics":[["/x","pkg/msg/T"],...]}
//   text   {"op":"subscribed","topic":"/x","type":"...","kind":"image|cloud|json"}
//   text   {"op":"msg","topic":"/x","data":{...}}           (kind json)
//   text   {"op":"error","topic":"/x","error":"..."}
//   text   {"op":"result","success":true,"message":"..."}   (calibrate, clear_leader)
//   text   {"op":"recording","state":"idle|recording|saving","name":"...","elapsed":s,"error":"..."}
//   binary [u8 kind][u8 0][u16le topic_len][topic][pad to 4][payload]
//          kind 1 = JPEG, 2 = PNG, 3 = float32 x,y,z,intensity points
//          kind 1 and 2 start the payload with [u32 sec][u32 nanosec] of the
//          image header stamp

#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

#include <spawn.h>
#include <sys/wait.h>

#include <ament_index_cpp/get_package_share_path.hpp>
#include <follow_anywhere_interfaces/srv/perception_calibrate_leader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "encoders.hpp"
#include "msg_json.hpp"
#include "server.hpp"

using namespace std::chrono_literals;

extern char ** environ;

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

bool has_field(const std::string & js, const std::string & key)
{
  return js.find("\"" + key + "\"") != std::string::npos;
}

// Header stamp of a message that starts with std_msgs/Header, read straight
// from the CDR bytes (4 byte encapsulation, then int32 sec, uint32 nanosec),
// as sec << 32 | nanosec. 0 when the message is too short.
uint64_t header_stamp(const rclcpp::SerializedMessage & raw)
{
  const auto & m = raw.get_rcl_serialized_message();
  if (m.buffer_length < 12) {return 0;}
  uint32_t sec, nsec;
  std::memcpy(&sec, m.buffer + 4, 4);
  std::memcpy(&nsec, m.buffer + 8, 4);
  if (!(m.buffer[1] & 1)) {  // big endian CDR
    sec = __builtin_bswap32(sec);
    nsec = __builtin_bswap32(nsec);
  }
  return (static_cast<uint64_t>(sec) << 32) | nsec;
}

Payload text(std::string s) {return std::make_shared<const std::string>(std::move(s));}

Payload binary_frame(
  uint8_t kind, const std::string & topic, const void * data, size_t size, uint64_t stamp = 0)
{
  size_t head = 4 + topic.size();
  size_t pad = (4 - head % 4) % 4;
  auto out = std::make_shared<std::string>();
  out->reserve(head + pad + 8 + size);
  out->push_back(static_cast<char>(kind));
  out->push_back(0);
  out->push_back(static_cast<char>(topic.size() & 0xff));
  out->push_back(static_cast<char>((topic.size() >> 8) & 0xff));
  out->append(topic);
  out->append(pad, '\0');
  if (kind == 1 || kind == 2) {
    uint32_t t[2] = {static_cast<uint32_t>(stamp >> 32), static_cast<uint32_t>(stamp)};
    out->append(reinterpret_cast<const char *>(t), sizeof(t));
  }
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

std::string clean_title(std::string title)
{
  for (auto & c : title) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {c = '_';}
  }
  return title.empty() ? "bag" : title;
}

std::filesystem::path free_path(const std::filesystem::path & dir, const std::string & name)
{
  auto path = dir / name;
  for (int i = 2; std::filesystem::exists(path); ++i) {
    path = dir / (name + "_" + std::to_string(i));
  }
  return path;
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
    pair_buffer_ = declare_parameter<int>("pair_buffer", 30);
    img_opt_.jpeg_quality = declare_parameter<int>("jpeg_quality", 70);
    img_opt_.max_width = declare_parameter<int>("max_width", 960);
    img_opt_.depth_max_m = declare_parameter<double>("depth_max_m", 6.0);
    defaults_ = declare_parameter<std::vector<std::string>>(
      "default_topics",
      {"/camera/camera/color/image_raw", "/utlidar/cloud_base", "/sportmodestate"});
    bag_directory_ = declare_parameter<std::string>("bag_directory", WEB_VIEWER_BAG_DIR);
    exclude_regex_ = declare_parameter<std::string>(
      "record_exclude_regex",
      ".*/image_raw$|.*/image_rect_raw$|.*/theora$|.*/zstd$|^/config_change_status$");

    if (www.empty()) {
      www = (ament_index_cpp::get_package_share_path("web_viewer") / "www").string();
    }

    ServerCallbacks cb;
    cb.on_open = [this](const SessionPtr & s) {on_open(s);};
    cb.on_text = [this](const SessionPtr & s, const std::string & m) {on_text(s, m);};
    cb.on_close = [this](const SessionPtr & s) {on_close(s);};
    calibrate_client_ = create_client<follow_anywhere_interfaces::srv::PerceptionCalibrateLeader>(
      "/perception/calibrate_leader");
    clear_client_ = create_client<std_srvs::srv::Trigger>("/perception/clear_leader");
    // Output topics
    debug_pub_ = create_publisher<std_msgs::msg::String>("/web_viewer/debug", 10);

    server_ = std::make_unique<Server>(address, static_cast<uint16_t>(port), www, cb);
    server_->start();

    topics_timer_ = create_wall_timer(2s, [this] {refresh_topics(false);});
    RCLCPP_INFO(get_logger(), "web_viewer on http://%s:%ld (files from %s)",
      address.c_str(), port, www.c_str());
  }

  ~WebViewer() override
  {
    {
      std::lock_guard<std::mutex> lk(record_mtx_);
      if (record_state_ == "recording") {
        record_state_ = "saving";
        kill(recorder_pid_, SIGINT);
      }
    }
    if (recorder_thread_.joinable()) {recorder_thread_.join();}
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
    double hz = -1;  // < 0: the default of its kind
    // Paired image stream: images wait here until a message on `pair` has
    // the same stamp. Guarded by pair_mtx (and mtx_ when written).
    std::mutex pair_mtx;
    std::string pair;
    std::deque<std::pair<uint64_t, std::shared_ptr<const rclcpp::SerializedMessage>>> held;
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
    s->send(recording_state(), false, false);
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
    if (op == "calibrate") {
      auto id = json_field(msg, "track_id");
      calibrate(s, id.empty() ? -1 : static_cast<int32_t>(std::strtol(id.c_str(), nullptr, 10)));
      return;
    }
    if (op == "clear_leader") {
      clear_leader(s);
      return;
    }
    if (op == "record_start") {
      start_recording();
      return;
    }
    if (op == "record_stop") {
      stop_recording(json_field(msg, "title"));
      return;
    }
    auto topic = json_field(msg, "topic");
    if (topic.empty()) {return;}
    if (op == "sub") {
      std::optional<double> hz;
      std::optional<std::string> pair;
      if (has_field(msg, "hz")) {hz = std::strtod(json_field(msg, "hz").c_str(), nullptr);}
      if (has_field(msg, "pair")) {pair = json_field(msg, "pair");}
      subscribe(s, topic, hz, pair);
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

  // ------------------------------------------------------------ services

  void calibrate(const SessionPtr & s, int32_t track_id)
  {
    using Srv = follow_anywhere_interfaces::srv::PerceptionCalibrateLeader;
    if (!calibrate_client_->service_is_ready()) {
      send_result(s, false, "/perception/calibrate_leader is not available");
      return;
    }
    auto req = std::make_shared<Srv::Request>();
    req->track_id = track_id;
    std::weak_ptr<WsSession> weak = s;
    calibrate_client_->async_send_request(req, [this, weak](rclcpp::Client<Srv>::SharedFuture f) {
        if (auto sp = weak.lock()) {send_result(sp, f.get()->success, f.get()->message);}
      });
  }

  void clear_leader(const SessionPtr & s)
  {
    using Srv = std_srvs::srv::Trigger;
    if (!clear_client_->service_is_ready()) {
      send_result(s, false, "/perception/clear_leader is not available");
      return;
    }
    std::weak_ptr<WsSession> weak = s;
    clear_client_->async_send_request(std::make_shared<Srv::Request>(),
      [this, weak](rclcpp::Client<Srv>::SharedFuture f) {
        if (auto sp = weak.lock()) {send_result(sp, f.get()->success, f.get()->message);}
      });
  }

  void send_result(const SessionPtr & s, bool success, const std::string & message)
  {
    std::string m = "{\"op\":\"result\",\"success\":";
    m += success ? "true" : "false";
    m += ",\"message\":";
    json_escape(m, message);
    m += '}';
    s->send(text(m), false, false);
  }

  // ----------------------------------------------------------- recording

  void start_recording()
  {
    std::string error;
    {
      std::lock_guard<std::mutex> lk(record_mtx_);
      if (record_state_ != "idle") {return;}
      if (recorder_thread_.joinable()) {recorder_thread_.join();}

      char stamp[16];
      auto now = std::time(nullptr);
      std::strftime(stamp, sizeof(stamp), "%m%d_%H%M", std::localtime(&now));
      std::error_code ec;
      std::filesystem::create_directories(bag_directory_, ec);
      auto path = free_path(bag_directory_, std::string("recording_") + stamp);

      std::vector<std::string> args =
      {"ros2", "bag", "record", "-a", "-o", path.string(), "--exclude-regex", exclude_regex_};
      std::vector<char *> argv;
      for (auto & a : args) {argv.push_back(a.data());}
      argv.push_back(nullptr);
      posix_spawnattr_t attr;
      posix_spawnattr_init(&attr);
      posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
      posix_spawnattr_setpgroup(&attr, 0);
      pid_t pid;
      int err = posix_spawnp(&pid, "ros2", nullptr, &attr, argv.data(), environ);
      posix_spawnattr_destroy(&attr);

      if (err != 0) {
        error = std::string("cannot start ros2 bag record: ") + std::strerror(err);
        record_error_ = error;
      } else {
        recorder_pid_ = pid;
        record_path_ = path;
        record_stamp_ = stamp;
        record_name_ = path.filename().string();
        record_title_.clear();
        record_error_.clear();
        record_start_ = std::chrono::steady_clock::now();
        record_state_ = "recording";
        recorder_thread_ = std::thread([this, pid] {wait_for_recorder(pid);});
        RCLCPP_INFO(get_logger(), "recording bag to %s", path.c_str());
      }
    }
    if (!error.empty()) {report_record_error(error);}
    broadcast(recording_state());
  }

  void stop_recording(const std::string & title)
  {
    {
      std::lock_guard<std::mutex> lk(record_mtx_);
      if (record_state_ != "recording") {return;}
      record_title_ = clean_title(title);
      record_state_ = "saving";
      kill(recorder_pid_, SIGINT);
    }
    broadcast(recording_state());
  }

  void wait_for_recorder(pid_t pid)
  {
    int status = 0;
    waitpid(pid, &status, 0);
    std::string error;
    {
      std::lock_guard<std::mutex> lk(record_mtx_);
      if (record_state_ != "saving") {
        error = "ros2 bag record stopped by itself (" + std::string(WIFSIGNALED(status) ?
          "signal " + std::to_string(WTERMSIG(status)) :
          "exit code " + std::to_string(WEXITSTATUS(status))) + "), bag left as " + record_name_;
      } else if (!std::filesystem::exists(record_path_ / "metadata.yaml")) {
        error = "bag was not finalised, no metadata.yaml in " + record_name_;
      } else if (!record_title_.empty()) {
        auto target = free_path(bag_directory_, record_title_ + "_" + record_stamp_);
        std::error_code ec;
        std::filesystem::rename(record_path_, target, ec);
        if (ec) {
          error = "cannot rename " + record_name_ + ": " + ec.message();
        } else {
          record_name_ = target.filename().string();
        }
      }
      record_state_ = "idle";
      record_error_ = error;
      recorder_pid_ = -1;
      if (error.empty()) {RCLCPP_INFO(get_logger(), "saved bag %s", record_name_.c_str());}
    }
    if (!error.empty()) {report_record_error(error);}
    broadcast(recording_state());
  }

  void report_record_error(const std::string & error)
  {
    RCLCPP_ERROR(get_logger(), "%s", error.c_str());
    std_msgs::msg::String m;
    m.data = "BagRecorder: " + error;
    debug_pub_->publish(m);
  }

  Payload recording_state()
  {
    std::lock_guard<std::mutex> lk(record_mtx_);
    auto elapsed = record_state_ == "idle" ? 0.0 :
      std::chrono::duration<double>(std::chrono::steady_clock::now() - record_start_).count();
    std::string m = "{\"op\":\"recording\",\"state\":\"" + record_state_ + "\",\"name\":";
    json_escape(m, record_name_);
    m += ",\"elapsed\":" + std::to_string(elapsed) + ",\"error\":";
    json_escape(m, record_error_);
    m += '}';
    return text(std::move(m));
  }

  void broadcast(const Payload & p)
  {
    std::vector<SessionPtr> targets;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      for (auto & [id, w] : sessions_) {
        if (auto s = w.lock()) {targets.push_back(s);}
      }
    }
    for (auto & s : targets) {s->send(p, false, false);}
  }

  // ------------------------------------------------------- subscriptions

  void subscribe(
    const SessionPtr & s, const std::string & topic, std::optional<double> hz,
    std::optional<std::string> pair)
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
            if (auto sp = weak.lock()) {on_message(*sp, m);}
          }, opts);
      } catch (const std::exception & e) {
        send_error(s, topic, std::string("cannot subscribe: ") + e.what());
        return;
      }
      RCLCPP_INFO(get_logger(), "streaming %s [%s]", topic.c_str(), st->type.c_str());
      it = streams_.emplace(topic, st).first;
    }
    auto & st = *it->second;
    st.clients[s->id()] = s;
    if (hz) {st.hz = *hz;}
    if (pair && (st.kind == Kind::Image || st.kind == Kind::Compressed)) {
      std::lock_guard<std::mutex> plk(st.pair_mtx);
      st.pair = *pair;
      st.held.clear();
    }

    std::string m = "{\"op\":\"subscribed\",\"topic\":";
    json_escape(m, topic);
    m += ",\"type\":";
    json_escape(m, it->second->type);
    m += ",\"kind\":\"";
    m += kind_name(it->second->kind);
    m += "\"}";
    s->send(text(m), false, false);
  }

  void on_message(Stream & st, const std::shared_ptr<const rclcpp::SerializedMessage> & raw)
  {
    {
      std::lock_guard<std::mutex> plk(st.pair_mtx);
      if (!st.pair.empty()) {
        st.held.emplace_back(header_stamp(*raw), raw);
        while (st.held.size() > static_cast<size_t>(pair_buffer_)) {st.held.pop_front();}
        return;
      }
    }

    double hz = st.hz >= 0 ? st.hz : st.kind == Kind::Json ? json_hz_ :
      (st.kind == Kind::Cloud || st.kind == Kind::Scan) ? cloud_hz_ : image_hz_;
    auto now = std::chrono::steady_clock::now();
    if (hz > 0 && now - st.last < std::chrono::duration<double>(1.0 / hz)) {return;}
    st.last = now;

    send_paired_images(st.topic, header_stamp(*raw));
    publish(st, *raw);
  }

  // Images paired to `topic` that have this stamp are sent now, so they reach
  // the browser just before the message they belong to.
  void send_paired_images(const std::string & topic, uint64_t stamp)
  {
    std::vector<std::shared_ptr<Stream>> paired;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      for (auto & [name, st] : streams_) {
        if (st->pair == topic) {paired.push_back(st);}
      }
    }
    for (auto & st : paired) {
      std::shared_ptr<const rclcpp::SerializedMessage> image;
      {
        std::lock_guard<std::mutex> plk(st->pair_mtx);
        auto & held = st->held;
        for (auto it = held.begin(); it != held.end(); ++it) {
          if (it->first == stamp) {
            image = it->second;
            held.erase(held.begin(), std::next(it));
            break;
          }
        }
      }
      if (image) {publish(*st, *image);}
    }
  }

  void publish(Stream & st, const rclcpp::SerializedMessage & raw)
  {
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
            out = binary_frame(1, st.topic, jpg.data(), jpg.size(), header_stamp(raw));
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
            out = binary_frame(k, st.topic, img.data.data(), img.data.size(), header_stamp(raw));
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
    for (auto & s : clients) {s->send(out, is_binary, true, st.topic);}
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
  int64_t max_points_, max_array_, pair_buffer_;
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
  rclcpp::Client<follow_anywhere_interfaces::srv::PerceptionCalibrateLeader>::SharedPtr
    calibrate_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr clear_client_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;

  std::string bag_directory_, exclude_regex_;
  std::mutex record_mtx_;
  std::string record_state_ = "idle";
  std::string record_name_, record_title_, record_stamp_, record_error_;
  std::filesystem::path record_path_;
  std::chrono::steady_clock::time_point record_start_;
  pid_t recorder_pid_ = -1;
  std::thread recorder_thread_;

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
