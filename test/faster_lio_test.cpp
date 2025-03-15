//
// Created by myx on 2025/3/9.
//

#include "lio_base.h"
#include "faster_lio.h"

#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>


int main(int argc, char **argv) {
    ros::init(argc, argv, "lio_bp_iekf");
    ros::NodeHandle nh;

    // 初始化glog库
    google::InitGoogleLogging(argv[0]);

    // 设置日志输出目录
    FLAGS_log_dir = "./log";

    // 设置日志级别，INFO级别及以上的日志会被输出
    FLAGS_minloglevel = google::INFO;

    auto lio_mapping = std::make_shared<simple_lio::faster_lio>();
    lio_mapping->Init(nh);
    lio_mapping->SetLIOProcessFunc([&](const simple_lio::LIOOutput & lio_output){
      // Here you can define yourself processing function.
      ROS_INFO_ONCE("Into lio process function");
    });

    ros::AsyncSpinner spinner(32);
    spinner.start();

    ros::Rate rate(5000);
    while (ros::ok()) {

      lio_mapping->Run();
      rate.sleep();
    }

    spinner.stop();

  return 0;
}