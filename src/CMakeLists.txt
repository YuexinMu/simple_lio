add_library(${PROJECT_NAME}
        lio.cpp
        common/pointcloud_preprocess.cpp
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