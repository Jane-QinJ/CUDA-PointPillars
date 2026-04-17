#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>

#include "common/check.hpp"
#include "pointpillar.hpp"

namespace {

bool hasField(const sensor_msgs::PointCloud2& msg, const std::string& field_name) {
  for (const auto& field : msg.fields) {
    if (field.name == field_name) {
      return true;
    }
  }
  return false;
}

geometry_msgs::Quaternion yawToQuaternion(float yaw) {
  geometry_msgs::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw * 0.5f);
  q.w = std::cos(yaw * 0.5f);
  return q;
}

std_msgs::ColorRGBA classColor(int id) {
  std_msgs::ColorRGBA color;
  color.a = 0.55f;
  switch (id) {
    case 0:
      color.r = 0.20f;
      color.g = 0.85f;
      color.b = 0.35f;
      break;
    case 1:
      color.r = 0.95f;
      color.g = 0.75f;
      color.b = 0.20f;
      break;
    default:
      color.r = 0.15f;
      color.g = 0.55f;
      color.b = 0.95f;
      break;
  }
  return color;
}

pointpillar::lidar::VoxelizationParameter makeVoxelizationParam() {
  pointpillar::lidar::VoxelizationParameter vp;
  vp.min_range = nvtype::Float3(0.0f, -39.68f, -3.0f);
  vp.max_range = nvtype::Float3(69.12f, 39.68f, 1.0f);
  vp.voxel_size = nvtype::Float3(0.16f, 0.16f, 4.0f);
  vp.grid_size =
      vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
  vp.max_voxels = 40000;
  vp.max_points_per_voxel = 32;
  vp.max_points = 300000;
  vp.num_feature = 4;
  return vp;
}

pointpillar::lidar::PostProcessParameter makePostParam(
    const pointpillar::lidar::VoxelizationParameter& vp) {
  pointpillar::lidar::PostProcessParameter pp;
  pp.min_range = vp.min_range;
  pp.max_range = vp.max_range;
  pp.feature_size = nvtype::Int2(vp.grid_size.x / 2, vp.grid_size.y / 2);
  return pp;
}

class PointPillarRosNode {
 public:
  PointPillarRosNode() : nh_(), pnh_("~") {
    std::string package_path = ros::package::getPath("pointpillar");
    std::string default_model =
        package_path.empty() ? "model/pointpillar.plan"
                             : package_path + "/model/pointpillar.plan";

    pnh_.param<std::string>("model_file", model_file_, default_model);
    pnh_.param<std::string>("points_topic", points_topic_, "/points_raw");
    pnh_.param<std::string>("marker_topic", marker_topic_, "/pointpillar/markers");
    pnh_.param<bool>("enable_timer", enable_timer_, false);
    pnh_.param<int>("queue_size", queue_size_, 1);

    auto vp = makeVoxelizationParam();
    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp;
    param.lidar_model = model_file_;
    param.lidar_post = makePostParam(vp);

    core_ = pointpillar::lidar::create_core(param);
    if (!core_) {
      throw std::runtime_error("Failed to create PointPillars core. Check model_file.");
    }

    core_->set_timer(enable_timer_);
    core_->print();
    checkRuntime(cudaStreamCreate(&stream_));

    marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 1);
    points_sub_ = nh_.subscribe(points_topic_, queue_size_,
                                &PointPillarRosNode::pointsCallback, this);

    ROS_INFO_STREAM("PointPillars ROS node ready."
                    << " model_file=" << model_file_
                    << " points_topic=" << points_topic_
                    << " marker_topic=" << marker_topic_);
  }

  ~PointPillarRosNode() {
    if (stream_ != nullptr) {
      checkRuntime(cudaStreamDestroy(stream_));
      stream_ = nullptr;
    }
  }

 private:
  void pointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!hasField(*msg, "x") || !hasField(*msg, "y") || !hasField(*msg, "z")) {
      ROS_WARN_THROTTLE(5.0, "PointCloud2 is missing x/y/z fields.");
      return;
    }

    const bool has_intensity = hasField(*msg, "intensity");
    std::vector<float> points;
    points.reserve(static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height) * 4);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    if (has_intensity) {
      sensor_msgs::PointCloud2ConstIterator<float> iter_i(*msg, "intensity");
      for (; iter_x != iter_x.end();
           ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
        if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) ||
            !std::isfinite(*iter_z)) {
          continue;
        }
        points.push_back(*iter_x);
        points.push_back(*iter_y);
        points.push_back(*iter_z);
        points.push_back(*iter_i);
      }
    } else {
      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) ||
            !std::isfinite(*iter_z)) {
          continue;
        }
        points.push_back(*iter_x);
        points.push_back(*iter_y);
        points.push_back(*iter_z);
        points.push_back(0.0f);
      }
    }

    if (points.empty()) {
      ROS_WARN_THROTTLE(5.0, "Received an empty or invalid point cloud.");
      publishDeleteAll(msg->header);
      return;
    }

    auto boxes = core_->forward(points.data(), static_cast<int>(points.size() / 4), stream_);
    publishBoxes(boxes, msg->header);
  }

  void publishDeleteAll(const std_msgs::Header& header) {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.action = visualization_msgs::Marker::DELETEALL;

    visualization_msgs::MarkerArray array;
    array.markers.push_back(marker);
    marker_pub_.publish(array);
  }

  void publishBoxes(const std::vector<pointpillar::lidar::BoundingBox>& boxes,
                    const std_msgs::Header& header) {
    visualization_msgs::MarkerArray array;

    visualization_msgs::Marker clear_marker;
    clear_marker.header = header;
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    array.markers.push_back(clear_marker);

    int marker_id = 0;
    for (const auto& box : boxes) {
      visualization_msgs::Marker marker;
      marker.header = header;
      marker.ns = "pointpillar_boxes";
      marker.id = marker_id++;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = box.x;
      marker.pose.position.y = box.y;
      marker.pose.position.z = box.z;
      marker.pose.orientation = yawToQuaternion(box.rt);
      marker.scale.x = std::max(box.l, 0.01f);
      marker.scale.y = std::max(box.w, 0.01f);
      marker.scale.z = std::max(box.h, 0.01f);
      marker.color = classColor(box.id);
      marker.lifetime = ros::Duration(0.1);
      array.markers.push_back(marker);
    }

    marker_pub_.publish(array);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber points_sub_;
  ros::Publisher marker_pub_;

  std::shared_ptr<pointpillar::lidar::Core> core_;
  cudaStream_t stream_ = nullptr;
  std::mutex mutex_;

  std::string model_file_;
  std::string points_topic_;
  std::string marker_topic_;
  bool enable_timer_ = false;
  int queue_size_ = 1;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "pointpillar_ros_node");

  try {
    PointPillarRosNode node;
    ros::spin();
  } catch (const std::exception& e) {
    ROS_FATAL_STREAM(e.what());
    return 1;
  }

  return 0;
}
