cmake_minimum_required(VERSION 2.8.3)
project(plan_env)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(OpenCV REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
  cv_bridge
  message_filters
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

add_message_files(
  FILES
  MultipleMasksWithConfidence.msg
  SemanticObservation.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES plan_env
  CATKIN_DEPENDS roscpp std_msgs message_runtime sensor_msgs
  #  DEPENDS system_lib
)

include_directories(
  SYSTEM
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})


add_library(plan_env
  src/sdf_map2d.cpp
  src/map_ros.cpp
  src/raycast2d.cpp
  src/perception_utils2d.cpp
  src/frontier_map2d.cpp
  src/object_map2d.cpp
  src/value_map2d.cpp
  src/region_graph2d.cpp
)
target_link_libraries(plan_env
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

if(CATKIN_ENABLE_TESTING)
  add_executable(region_graph_dirty_fixture test/region_graph_dirty_fixture.cpp)
  target_link_libraries(region_graph_dirty_fixture plan_env ${catkin_LIBRARIES})
  add_executable(region_graph2d_fixture test/region_graph2d_fixture.cpp)
  target_link_libraries(region_graph2d_fixture plan_env ${catkin_LIBRARIES})
endif()
