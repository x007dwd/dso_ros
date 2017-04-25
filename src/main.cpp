/**
* This file is part of DSO.
*
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "util/settings.h"
//#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

#include "cv_bridge/cv_bridge.h"
#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
bool cameraInfoRdyFlag = false;
using namespace dso;
void CamInfoCbk() {}
void parseArgument(char *arg) {
  int option;
  char buf[1000];

  if (1 == sscanf(arg, "option=%d", &option)) {
    benchmarkSpecialOption = option;
    printf("OPTION %d!!\n", benchmarkSpecialOption);
    return;
  }

  if (1 == sscanf(arg, "nolog=%d", &option)) {
    if (option == 1) {
      setting_logStuff = false;
      printf("DISABLE LOGGING!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "nogui=%d", &option)) {
    if (option == 1) {
      disableAllDisplay = true;
      printf("NO GUI!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nomt=%d", &option)) {
    if (option == 1) {
      multiThreading = false;
      printf("NO MultiThreading!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "calib=%s", buf)) {
    calib = buf;
    printf("loading calibration from %s!\n", calib.c_str());
    return;
  }
  if (1 == sscanf(arg, "vignette=%s", buf)) {
    vignetteFile = buf;
    printf("loading vignette from %s!\n", vignetteFile.c_str());
    return;
  }

  if (1 == sscanf(arg, "gamma=%s", buf)) {
    gammaFile = buf;
    printf("loading gammaCalib from %s!\n", gammaFile.c_str());
    return;
  }

  printf("could not parse argument \"%s\"!!\n", arg);
}

FullSystem *fullSystem = 0;
Undistort *undistorter = 0;
int frameID = 0;

void vidCb(const sensor_msgs::ImageConstPtr img) {
  cv_bridge::CvImagePtr cv_ptr =
      cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  assert(cv_ptr->image.type() == CV_8U);
  assert(cv_ptr->image.channels() == 1);

  if (setting_fullResetRequested) {
    // IOWrap::Output3DWrapper* wrap = fullSystem->outputWrapper;
    delete fullSystem;
    // if(wrap != 0) wrap->reset();
    fullSystem = new FullSystem();
    fullSystem->linearizeOperation = false;
    // fullSystem->outputWrapper = wrap;
    if (undistorter->photometricUndist != 0)
      fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
    setting_fullResetRequested = false;
  }

  MinimalImageB minImg((int)cv_ptr->image.cols, (int)cv_ptr->image.rows,
                       (unsigned char *)cv_ptr->image.data);
  ImageAndExposure *undistImg =
      undistorter->undistort<unsigned char>(&minImg, 1, 0, 1.0f);
  fullSystem->addActiveFrame(undistImg, frameID);
  frameID++;
  delete undistImg;
}
void cameraInfoCbk(const sensor_msgs::CameraInfoPtr &msg) {
  if (cameraInfoRdyFlag == false) {
    cameraInfoRdyFlag = true;
    Eigen::Matrix3f KMat;
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        KMat(i, j) = msg->K[3 * i + j];
      }
    }
    setGlobalCalib(msg->width, msg->height, KMat);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "dso_live");
  ros::NodeHandle nh;

  setlocale(LC_ALL, "");
  for (int i = 1; i < argc; i++)
    parseArgument(argv[i]);

  setting_desiredImmatureDensity = 1000;
  setting_desiredPointDensity = 1200;
  setting_minFrames = 5;
  setting_maxFrames = 7;
  setting_maxOptIterations = 4;
  setting_minOptIterations = 1;
  setting_logStuff = false;
  setting_kfGlobalWeight = 1.3;

  printf("MODE WITH CALIBRATION, but without exposure times!\n");
  setting_photometricCalibration = 0;
  setting_affineOptModeA = 0;
  setting_affineOptModeB = 0;

  ros::Subscriber camera_infoSub =
      nh.subscribe("/svo/camera_info", 10, cameraInfoCbk);
  // undistorter = Undistort::getUndistorterForFile(calib, gammaFile,
  // vignetteFile);

  // setGlobalCalib((int)undistorter->getSize()[0],
  //                (int)undistorter->getSize()[1],
  //                undistorter->getK().cast<float>());

  fullSystem = new FullSystem();
  fullSystem->linearizeOperation = false;
  /*    fullSystem->outputWrapper = new IOWrap::PangolinDSOViewer(
                   (int)undistorter->getSize()[0],
                   (int)undistorter->getSize()[1]);
  */

  ros::Subscriber imgSub = nh.subscribe("image", 1, &vidCb);

  ros::spin();

  delete undistorter;
  delete fullSystem;

  return 0;
}
