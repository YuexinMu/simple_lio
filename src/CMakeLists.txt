add_library(${PROJECT_NAME}
        faster_lio.cpp
        lio_base.cpp
        lio.cc
        common/pointcloud_preprocess.cpp
#        common/imu_processing.cpp
        common/utils.cpp
        )

add_dependencies(${PROJECT_NAME} livox_ros_driver_gencpp)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${g2o_libs}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${CSPARSE_LIBRARY}
        ${CHOLMOD_LIBRARY}
        ${CHOLMOD_LIBRARIES}
        tbb
        glog
        yaml-cpp
        )

target_include_directories(${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})

add_executable(run_lio_mapping run_lio_mapping.cc)
target_link_libraries(run_lio_mapping
        ${PROJECT_NAME} gflags
        )