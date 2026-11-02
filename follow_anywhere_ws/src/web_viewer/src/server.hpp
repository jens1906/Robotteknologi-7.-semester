// HTTP (static files) + WebSocket server on one port, built on Boost.Beast.
// Runs its own io_context threads; all callbacks are called from those threads.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/websocket/stream.hpp>

namespace web_viewer
{

class WsSession;
using SessionPtr = std::shared_ptr<WsSession>;
using Payload = std::shared_ptr<const std::string>;

struct ServerCallbacks
{
  std::function<void(const SessionPtr &)> on_open;
  std::function<void(const SessionPtr &, const std::string &)> on_text;
  std::function<void(const SessionPtr &)> on_close;
};

class WsSession : public std::enable_shared_from_this<WsSession>
{
public:
  WsSession(boost::asio::ip::tcp::socket && socket, const ServerCallbacks & cb, size_t max_queue);

  template<class Request>
  void run(Request req);

  // Thread safe. Droppable messages are skipped while the client is behind,
  // so a slow browser gets a lower frame rate instead of growing memory. A
  // droppable message with a key replaces a waiting one with the same key,
  // so the browser always gets the newest frame of a topic, not an old one.
  void send(Payload data, bool binary, bool droppable, std::string key = {});
  uint64_t id() const {return id_;}

private:
  void do_read();
  void do_write();
  void close();

  struct Out
  {
    Payload data;
    bool binary;
    bool droppable;
    std::string key;
  };

  boost::beast::websocket::stream<boost::beast::tcp_stream> ws_;
  boost::beast::flat_buffer buffer_;
  std::deque<Out> queue_;
  const ServerCallbacks & cb_;
  size_t max_queue_;
  uint64_t id_;
  bool open_ = false;
  bool closed_ = false;
};

class Server
{
public:
  Server(
    const std::string & address, uint16_t port, std::string www_root,
    ServerCallbacks cb, size_t threads = 2, size_t max_queue = 8);
  ~Server();

  void start();
  void stop();

private:
  void do_accept();

  boost::asio::io_context ioc_;
  boost::asio::ip::tcp::acceptor acceptor_;
  std::string www_root_;
  ServerCallbacks cb_;
  size_t n_threads_;
  size_t max_queue_;
  std::vector<std::thread> threads_;
};

}  // namespace web_viewer
