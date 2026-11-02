#include "server.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include <boost/asio/dispatch.hpp>
#include <boost/asio/post.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;

namespace web_viewer
{

namespace
{

std::atomic<uint64_t> g_next_id{1};

std::string mime_type(const std::string & path)
{
  auto ext = path.substr(path.find_last_of('.') == std::string::npos ? path.size() :
      path.find_last_of('.'));
  if (ext == ".html") {return "text/html; charset=utf-8";}
  if (ext == ".js") {return "text/javascript; charset=utf-8";}
  if (ext == ".css") {return "text/css; charset=utf-8";}
  if (ext == ".json") {return "application/json";}
  if (ext == ".png") {return "image/png";}
  if (ext == ".svg") {return "image/svg+xml";}
  if (ext == ".ico") {return "image/x-icon";}
  return "application/octet-stream";
}

// One plain HTTP connection: serves files from www_root, or hands the socket
// to a WsSession when the request is a WebSocket upgrade.
class HttpSession : public std::enable_shared_from_this<HttpSession>
{
public:
  HttpSession(
    tcp::socket && socket, const std::string & root, const ServerCallbacks & cb,
    size_t max_queue)
  : stream_(std::move(socket)), root_(root), cb_(cb), max_queue_(max_queue) {}

  void run()
  {
    net::dispatch(stream_.get_executor(), [self = shared_from_this()] {self->do_read();});
  }

private:
  void do_read()
  {
    req_ = {};
    stream_.expires_after(std::chrono::seconds(30));
    http::async_read(stream_, buffer_, req_,
      [self = shared_from_this()](beast::error_code ec, std::size_t) {
        if (ec) {
          self->stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
          return;
        }
        self->on_request();
      });
  }

  void on_request()
  {
    if (websocket::is_upgrade(req_)) {
      stream_.expires_never();
      std::make_shared<WsSession>(stream_.release_socket(), cb_, max_queue_)->run(std::move(req_));
      return;
    }

    auto res = std::make_shared<http::response<http::string_body>>();
    res->version(req_.version());
    res->keep_alive(req_.keep_alive());
    res->set(http::field::server, "web_viewer");

    std::string target(req_.target());
    target = target.substr(0, target.find('?'));
    if (req_.method() != http::verb::get && req_.method() != http::verb::head) {
      res->result(http::status::bad_request);
      res->body() = "unsupported method";
    } else if (target.empty() || target[0] != '/' || target.find("..") != std::string::npos) {
      res->result(http::status::bad_request);
      res->body() = "bad path";
    } else {
      if (target.back() == '/') {target += "index.html";}
      std::ifstream f(root_ + target, std::ios::binary);
      if (!f) {
        res->result(http::status::not_found);
        res->set(http::field::content_type, "text/plain");
        res->body() = "not found: " + target;
      } else {
        std::ostringstream ss;
        ss << f.rdbuf();
        res->result(http::status::ok);
        res->set(http::field::content_type, mime_type(target));
        res->set(http::field::cache_control, "no-cache");
        if (req_.method() == http::verb::get) {res->body() = ss.str();}
      }
    }
    res->prepare_payload();

    http::async_write(stream_, *res,
      [self = shared_from_this(), res](beast::error_code ec, std::size_t) {
        if (ec) {return;}
        if (!res->keep_alive()) {
          self->stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
          return;
        }
        self->do_read();
      });
  }

  beast::tcp_stream stream_;
  beast::flat_buffer buffer_;
  http::request<http::string_body> req_;
  const std::string & root_;
  const ServerCallbacks & cb_;
  size_t max_queue_;
};

}  // namespace

// ---------------------------------------------------------------- WsSession

WsSession::WsSession(tcp::socket && socket, const ServerCallbacks & cb, size_t max_queue)
: ws_(std::move(socket)), cb_(cb), max_queue_(max_queue), id_(g_next_id++) {}

template<class Request>
void WsSession::run(Request req)
{
  ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::server));
  ws_.write_buffer_bytes(1 << 16);
  ws_.async_accept(req, [self = shared_from_this()](beast::error_code ec) {
      if (ec) {return;}
      self->open_ = true;
      if (self->cb_.on_open) {self->cb_.on_open(self);}
      self->do_read();
      if (!self->queue_.empty()) {self->do_write();}
    });
}

void WsSession::send(Payload data, bool binary, bool droppable)
{
  net::post(ws_.get_executor(),
    [self = shared_from_this(), data = std::move(data), binary, droppable]() mutable {
      if (self->closed_) {return;}
      if (droppable && self->queue_.size() >= self->max_queue_) {return;}
      self->queue_.push_back({std::move(data), binary});
      if (self->open_ && self->queue_.size() == 1) {self->do_write();}
    });
}

void WsSession::do_read()
{
  ws_.async_read(buffer_, [self = shared_from_this()](beast::error_code ec, std::size_t) {
      if (ec) {
        self->close();
        return;
      }
      if (self->ws_.got_text() && self->cb_.on_text) {
        self->cb_.on_text(self, beast::buffers_to_string(self->buffer_.data()));
      }
      self->buffer_.consume(self->buffer_.size());
      self->do_read();
    });
}

void WsSession::do_write()
{
  auto & out = queue_.front();
  ws_.binary(out.binary);
  ws_.async_write(net::buffer(*out.data),
    [self = shared_from_this()](beast::error_code ec, std::size_t) {
      if (ec) {
        self->close();
        return;
      }
      self->queue_.pop_front();
      if (!self->queue_.empty()) {self->do_write();}
    });
}

void WsSession::close()
{
  if (closed_) {return;}
  closed_ = true;
  queue_.clear();
  if (open_ && cb_.on_close) {cb_.on_close(shared_from_this());}
}

// ------------------------------------------------------------------- Server

Server::Server(
  const std::string & address, uint16_t port, std::string www_root, ServerCallbacks cb,
  size_t threads, size_t max_queue)
: ioc_(static_cast<int>(threads)), acceptor_(net::make_strand(ioc_)),
  www_root_(std::move(www_root)), cb_(std::move(cb)), n_threads_(threads), max_queue_(max_queue)
{
  tcp::endpoint ep(net::ip::make_address(address), port);
  acceptor_.open(ep.protocol());
  acceptor_.set_option(net::socket_base::reuse_address(true));
  acceptor_.bind(ep);
  acceptor_.listen(net::socket_base::max_listen_connections);
}

Server::~Server() {stop();}

void Server::start()
{
  do_accept();
  for (size_t i = 0; i < n_threads_; ++i) {
    threads_.emplace_back([this] {ioc_.run();});
  }
}

void Server::stop()
{
  ioc_.stop();
  for (auto & t : threads_) {
    if (t.joinable()) {t.join();}
  }
  threads_.clear();
}

void Server::do_accept()
{
  acceptor_.async_accept(net::make_strand(ioc_), [this](beast::error_code ec, tcp::socket s) {
      if (!ec) {
        std::make_shared<HttpSession>(std::move(s), www_root_, cb_, max_queue_)->run();
      }
      do_accept();
    });
}

}  // namespace web_viewer
