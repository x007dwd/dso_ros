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

#pragma once
#include "IOWrapper/Output3DWrapper.h"
#include "boost/thread.hpp"

#include "util/MinimalImage.h"

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
namespace dso {

class FrameHessian;
class CalibHessian;
class FrameShell;

namespace IOWrap {

class ROSOutputWrapper : public Output3DWrapper {
public:
  inline ROSOutputWrapper(ros::NodeHandle nh) {
    printf("OUT: Created ROS OutputWrapper\n");

    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/dso/pose", 1);
  }

  virtual ~ROSOutputWrapper() { printf("OUT: Destroyed ROS OutputWrapper\n"); }

  virtual void
  publishGraph(const std::map<uint64_t, Eigen::Vector2i> &connectivity) {
    printf("OUT: got graph with %d edges\n", (int)connectivity.size());

    int maxWrite = 5;

    for (const std::pair<uint64_t, Eigen::Vector2i> &p : connectivity) {
      int idHost = p.first >> 32;
      int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
      printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n",
             idHost, idTarget, p.second[0], p.second[1]);
      maxWrite--;
      if (maxWrite == 0)
        break;
    }
  }

  virtual void publishKeyframes(std::vector<FrameHessian *> &frames, bool final,
                                CalibHessian *HCalib) {
    for (FrameHessian *f : frames) {
      printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d "
             "immature points. CameraToWorld:\n",
             f->frameID, final ? "final" : "non-final", f->shell->id,
             f->shell->timestamp, (int)f->pointHessians.size(),
             (int)f->pointHessiansMarginalized.size(),
             (int)f->immaturePoints.size());
      std::cout << f->shell->camToWorld.matrix3x4() << "\n";

      int maxWrite = 5;
      for (PointHessian *p : f->pointHessians) {
        printf("OUT: Example Point x=%.1f, y=%.1f, idepth=%f, idepth std.dev. "
               "%f, %d inlier-residuals\n",
               p->u, p->v, p->idepth_scaled, sqrt(1.0f / p->idepth_hessian),
               p->numGoodResiduals);
        maxWrite--;
        if (maxWrite == 0)
          break;
      }
    }
  }

  virtual void publishCamPose(FrameShell *frame, CalibHessian *HCalib) {
    printf("OUT: Current Frame %d (time %f, internal ID %d). CameraToWorld:\n",
           frame->id, frame->timestamp, frame->id);
    std::cout << frame->camToWorld.matrix3x4() << "\n";

    geometry_msgs::PoseStamped pose;
    pose.header.seq = frame->id;
    pose.header.stamp = ros::Time::now(); //(frame->timestamp);
    pose.header.frame_id = "dso";

    Eigen::Vector3d translation =
        frame->camToWorld.translation().cast<double>();
    pose.pose.position.x = translation.x();
    pose.pose.position.y = translation.y();
    pose.pose.position.z = translation.z();
    Eigen::Quaterniond quaternion =
        frame->camToWorld.unit_quaternion().cast<double>();
    pose.pose.orientation.w = quaternion.w();
    pose.pose.orientation.x = quaternion.x();
    pose.pose.orientation.y = quaternion.y();
    pose.pose.orientation.z = quaternion.z();
    pose_pub.publish(pose);
  }

  virtual void pushLiveFrame(FrameHessian *image) {
    // can be used to get the raw image / intensity pyramid.
  }

  virtual void pushDepthImage(MinimalImageB3 *image) {
    // can be used to get the raw image with depth overlay.
  }
  virtual bool needPushDepthImage() { return false; }

  virtual void pushDepthImageFloat(MinimalImageF *image, FrameHessian *KF) {
    printf("OUT: Predicted depth for KF %d (id %d, time %f, internal frame-ID "
           "%d). CameraToWorld:\n",
           KF->frameID, KF->shell->id, KF->shell->timestamp, KF->shell->id);
    std::cout << KF->shell->camToWorld.matrix3x4() << "\n";

    int maxWrite = 5;
    for (int y = 0; y < image->h; y++) {
      for (int x = 0; x < image->w; x++) {
        if (image->at(x, y) <= 0)
          continue;

        printf("OUT: Example Idepth at pixel (%d,%d): %f.\n", x, y,
               image->at(x, y));
        maxWrite--;
        if (maxWrite == 0)
          break;
      }
      if (maxWrite == 0)
        break;
    }
  }

private:
  ros::Publisher pose_pub;
};
}
}
