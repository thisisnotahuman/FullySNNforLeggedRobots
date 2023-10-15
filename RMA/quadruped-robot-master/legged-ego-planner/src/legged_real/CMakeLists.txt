cmake_minimum_required(VERSION 2.8.3)
project(legged_real)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    geometry_msgs
    quadrotor_msgs
    std_msgs
    # traj_utils
)

find_package(Eigen3)
catkin_package()

include_directories(
    include
    ${Boost_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-O3")


add_executable(trace src/trace.cpp)
target_link_libraries(trace ${EXTRA_LIBS} ${catkin_LIBRARIES})
add_dependencies(trace ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})