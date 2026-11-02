#include "msg_json.hpp"

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <rclcpp/typesupport_helpers.hpp>
#include <rosidl_typesupport_introspection_cpp/field_types.hpp>

namespace web_viewer
{

using rosidl_typesupport_introspection_cpp::MessageMember;
using rosidl_typesupport_introspection_cpp::MessageMembers;
namespace ft = rosidl_typesupport_introspection_cpp;

void json_escape(std::string & out, const std::string & s)
{
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

namespace
{

void put_double(std::string & out, double v)
{
  if (!std::isfinite(v)) {
    out += "null";
    return;
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", v);
  out += buf;
}

void write_message(std::string & out, const MessageMembers * mm, const void * msg, size_t max_array);

// One scalar at `p`. Arrays go through fetch_function into a temporary first,
// because std::vector<bool> has no addressable elements.
void write_scalar(std::string & out, const MessageMember & m, const void * p, size_t max_array)
{
  switch (m.type_id_) {
    case ft::ROS_TYPE_FLOAT: put_double(out, *static_cast<const float *>(p)); break;
    case ft::ROS_TYPE_DOUBLE: put_double(out, *static_cast<const double *>(p)); break;
    case ft::ROS_TYPE_LONG_DOUBLE:
      put_double(out, static_cast<double>(*static_cast<const long double *>(p))); break;
    case ft::ROS_TYPE_CHAR: out += std::to_string(*static_cast<const char *>(p)); break;
    case ft::ROS_TYPE_WCHAR: out += std::to_string(*static_cast<const char16_t *>(p)); break;
    case ft::ROS_TYPE_BOOLEAN: out += *static_cast<const bool *>(p) ? "true" : "false"; break;
    case ft::ROS_TYPE_OCTET:
    case ft::ROS_TYPE_UINT8: out += std::to_string(*static_cast<const uint8_t *>(p)); break;
    case ft::ROS_TYPE_INT8: out += std::to_string(*static_cast<const int8_t *>(p)); break;
    case ft::ROS_TYPE_UINT16: out += std::to_string(*static_cast<const uint16_t *>(p)); break;
    case ft::ROS_TYPE_INT16: out += std::to_string(*static_cast<const int16_t *>(p)); break;
    case ft::ROS_TYPE_UINT32: out += std::to_string(*static_cast<const uint32_t *>(p)); break;
    case ft::ROS_TYPE_INT32: out += std::to_string(*static_cast<const int32_t *>(p)); break;
    case ft::ROS_TYPE_UINT64: out += std::to_string(*static_cast<const uint64_t *>(p)); break;
    case ft::ROS_TYPE_INT64: out += std::to_string(*static_cast<const int64_t *>(p)); break;
    case ft::ROS_TYPE_STRING: json_escape(out, *static_cast<const std::string *>(p)); break;
    case ft::ROS_TYPE_WSTRING: {
        std::string s;
        for (char16_t c : *static_cast<const std::u16string *>(p)) {
          s += c < 0x80 ? static_cast<char>(c) : '?';
        }
        json_escape(out, s);
        break;
      }
    case ft::ROS_TYPE_MESSAGE:
      write_message(out, static_cast<const MessageMembers *>(m.members_->data), p, max_array);
      break;
    default: out += "null";
  }
}

void write_member(std::string & out, const MessageMember & m, const void * field, size_t max_array)
{
  if (!m.is_array_) {
    write_scalar(out, m, field, max_array);
    return;
  }
  size_t n = m.size_function ? m.size_function(field) : m.array_size_;
  size_t shown = std::min(n, max_array);
  out += '[';
  for (size_t i = 0; i < shown; ++i) {
    if (i) {out += ',';}
    if (m.type_id_ == ft::ROS_TYPE_MESSAGE || m.type_id_ == ft::ROS_TYPE_STRING ||
      m.type_id_ == ft::ROS_TYPE_WSTRING)
    {
      write_scalar(out, m, m.get_const_function(field, i), max_array);
    } else {
      alignas(16) unsigned char tmp[16];
      m.fetch_function(field, i, tmp);
      write_scalar(out, m, tmp, max_array);
    }
  }
  if (shown < n) {
    out += shown ? "," : "";
    out += "\"...";
    out += std::to_string(n - shown);
    out += " more\"";
  }
  out += ']';
}

void write_message(std::string & out, const MessageMembers * mm, const void * msg, size_t max_array)
{
  out += '{';
  for (uint32_t i = 0; i < mm->member_count_; ++i) {
    const auto & m = mm->members_[i];
    if (i) {out += ',';}
    json_escape(out, m.name_);
    out += ':';
    write_member(out, m, static_cast<const uint8_t *>(msg) + m.offset_, max_array);
  }
  out += '}';
}

}  // namespace

MsgToJson::MsgToJson(const std::string & type)
{
  cpp_lib_ = rclcpp::get_typesupport_library(type, "rosidl_typesupport_cpp");
  intro_lib_ = rclcpp::get_typesupport_library(type, "rosidl_typesupport_introspection_cpp");
  auto cpp_ts = rclcpp::get_message_typesupport_handle(type, "rosidl_typesupport_cpp", *cpp_lib_);
  auto intro_ts = rclcpp::get_message_typesupport_handle(
    type, "rosidl_typesupport_introspection_cpp", *intro_lib_);
  serializer_ = std::make_unique<rclcpp::SerializationBase>(cpp_ts);
  members_ = static_cast<const MessageMembers *>(intro_ts->data);
  if (!members_) {throw std::runtime_error("no introspection data for " + type);}
}

std::string MsgToJson::convert(const rclcpp::SerializedMessage & msg, size_t max_array) const
{
  std::vector<std::max_align_t> storage(members_->size_of_ / sizeof(std::max_align_t) + 1);
  void * buf = storage.data();
  members_->init_function(buf, rosidl_runtime_cpp::MessageInitialization::ALL);
  std::string out;
  try {
    serializer_->deserialize_message(&msg, buf);
    write_message(out, members_, buf, max_array);
  } catch (...) {
    members_->fini_function(buf);
    throw;
  }
  members_->fini_function(buf);
  return out;
}

}  // namespace web_viewer
