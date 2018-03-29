cmake_minimum_required(VERSION 2.8.3)
project(ros_speech2text)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

catkin_python_setup()

## ROS messages, services and actions

add_message_files(FILES transcript.msg event.msg status.msg)

generate_messages(DEPENDENCIES std_msgs)
catkin_package(
 CATKIN_DEPENDS rospy std_msgs message_runtime
)

## Build

include_directories(
  ${catkin_INCLUDE_DIRS}
)


## Test

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(test/test_s2t.test)
endif()

## Install

install(PROGRAMS
  scripts/ros_s2t.py
  ## scripts/ros_speech2text_sync.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
