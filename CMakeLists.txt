cmake_minimum_required(VERSION 3.0.2)
project(simple_lio)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-w")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -ggdb ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
add_subdirectory(thirdparty/livox_ros_driver)

include(cmake/packages.cmake)

add_subdirectory(src)

add_executable(run_lio_mapping test/run_lio_mapping.cc)
target_link_libraries(run_lio_mapping
        ${PROJECT_NAME} gflags
        )