// Turns a serialized message of any type into JSON, using the introspection
// type support that every ROS 2 message package installs.
#pragma once

#include <memory>
#include <string>

#include <rclcpp/serialized_message.hpp>
#include <rclcpp/serialization.hpp>
#include <rcpputils/shared_library.hpp>
#include <rosidl_typesupport_introspection_cpp/message_introspection.hpp>

namespace web_viewer
{

void json_escape(std::string & out, const std::string & s);

class MsgToJson
{
public:
  // Throws std::runtime_error when the type support can't be loaded.
  explicit MsgToJson(const std::string & type);

  // Arrays longer than max_array are cut and end with a "...N more" string,
  // so images and clouds don't flood the browser.
  std::string convert(const rclcpp::SerializedMessage & msg, size_t max_array) const;

private:
  std::shared_ptr<rcpputils::SharedLibrary> cpp_lib_, intro_lib_;
  std::unique_ptr<rclcpp::SerializationBase> serializer_;
  const rosidl_typesupport_introspection_cpp::MessageMembers * members_;
};

}  // namespace web_viewer
