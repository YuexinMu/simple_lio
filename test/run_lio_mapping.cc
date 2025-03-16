//
// Created by xiang on 2021/10/8.
//
#include <gflags/gflags.h>
#include <unistd.h>
#include <csignal>

#include "lio.h"


bool FLAG_EXIT = false;

void SigHandle(int sig) {
  FLAG_EXIT = true;
  ROS_WARN("catch sig %d", sig);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "lio_bp_iekf");
  ros::NodeHandle nh;

  signal(SIGINT, SigHandle);

  auto lio_mapping = std::make_shared<simple_lio::lio>();
  lio_mapping->InitROS(nh);

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

  lio_mapping->Finish();
  spinner.stop();


  return 0;
}
