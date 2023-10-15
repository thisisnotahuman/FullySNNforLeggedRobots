cmake_minimum_required(VERSION 3.0.2)
project(deeprobotics_legged_sdk)

add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "-O3")

# catkin_package(
#     INCLUDE_DIRS include
#     LIBRARIES ${PROJECT_NAME}
# )

# eigen3
find_package(Eigen3 REQUIRED)
include_directories(
    ${EIGEN3_INCLUDE_DIRS}
)
include_directories(include/deeprobotics_legged_sdk)

set(EXTRA_LIBS -lpthread -lm -lrt -ldl -lstdc++) # ${DEEPROBOTICS_SDK_LIB} lcm)
file(GLOB_RECURSE sources "src/*.cpp")
add_library(deeprobotics_legged_sdk SHARED ${sources})
target_link_libraries(deeprobotics_legged_sdk ${EXTRA_LIBS})
add_executable(example "main.cpp" ${SRC_LIST})
target_link_libraries(example deeprobotics_legged_sdk)


add_subdirectory(pybind11)
pybind11_add_module(lite3_interface python_interface.cpp)
target_link_libraries(lite3_interface PUBLIC deeprobotics_legged_sdk)