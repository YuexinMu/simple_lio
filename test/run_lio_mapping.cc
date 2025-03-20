//
// Created by xiang on 2021/10/8.
//
#include <gflags/gflags.h>
#include <unistd.h>
#include <csignal>
#include <filesystem>

#include "lio.h"


bool FLAG_EXIT = false;

void SigHandle(int sig) {
  FLAG_EXIT = true;
  ROS_WARN("catch sig %d", sig);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "lio_bp_iekf");
  ros::NodeHandle nh;

  std::string log_dir = std::string(ROOT_DIR) + "log";
  std::filesystem::path log_path(log_dir);
  // check the log path whether exist
  if (!std::filesystem::exists(log_path)) {
    // creat log path
    if (!std::filesystem::create_directories(log_path)) {
      std::cout << "Failed to create path: " << log_path << std::endl;
    }
  }

//  google::InitGoogleLogging(argv[0]);
//  FLAGS_log_dir = log_dir;
//  FLAGS_minloglevel = google::INFO;

  signal(SIGINT, SigHandle);

  auto lio_mapping = std::make_shared<simple_lio::lio>();
  lio_mapping->Init(nh);

  ros::AsyncSpinner spinner(32);
  spinner.start();

  ros::Rate rate(5000);
  while (ros::ok()) {
    if (FLAG_EXIT) {
      break;
    }
    lio_mapping->Run();
    rate.sleep();
  }

  lio_mapping->Finish(log_dir);
  spinner.stop();


  return 0;
}
