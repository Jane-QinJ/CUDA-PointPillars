#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/MarkerArray.h>

#include "common/check.hpp"
#include "pointpillar.hpp"
#include "pointpillar/Detection3D.h"
#include "pointpillar/Detection3DArray.h"

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

// Defaults match the KITTI 3-class cfg (cfgs/kitti_models/pointpillar.yaml).
// Override via ROS params below to match a custom-trained model's cfg
// without recompiling.
pointpillar::lidar::VoxelizationParameter makeVoxelizationParam(ros::NodeHandle& pnh) {
  pointpillar::lidar::VoxelizationParameter vp;

  std::vector<double> pc_range;
  if (!pnh.getParam("point_cloud_range", pc_range) || pc_range.size() != 6) {
    pc_range = {0.0, -39.68, -3.0, 69.12, 39.68, 1.0};
  }
  std::vector<double> voxel_size;
  if (!pnh.getParam("voxel_size", voxel_size) || voxel_size.size() != 3) {
    voxel_size = {0.16, 0.16, 4.0};
  }

  vp.min_range = nvtype::Float3(static_cast<float>(pc_range[0]),
                                static_cast<float>(pc_range[1]),
                                static_cast<float>(pc_range[2]));
  vp.max_range = nvtype::Float3(static_cast<float>(pc_range[3]),
                                static_cast<float>(pc_range[4]),
                                static_cast<float>(pc_range[5]));
  vp.voxel_size = nvtype::Float3(static_cast<float>(voxel_size[0]),
                                 static_cast<float>(voxel_size[1]),
                                 static_cast<float>(voxel_size[2]));
  vp.grid_size =
      vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
  vp.max_voxels = 40000;
  vp.max_points_per_voxel = 32;
  vp.max_points = 300000;
  vp.num_feature = 4;
  return vp;
}

pointpillar::lidar::PostProcessParameter makePostParam(
    const pointpillar::lidar::VoxelizationParameter& vp, ros::NodeHandle& pnh) {
  pointpillar::lidar::PostProcessParameter pp;
  pp.min_range = vp.min_range;
  pp.max_range = vp.max_range;
  pp.feature_size = nvtype::Int2(vp.grid_size.x / 2, vp.grid_size.y / 2);

  const int max_anchors = static_cast<int>(sizeof(pp.anchors) / sizeof(pp.anchors[0]) / 4);
  std::vector<double> anchors;
  if (pnh.getParam("anchors", anchors) && !anchors.empty()) {
    if (anchors.size() % 4 == 0 &&
        static_cast<int>(anchors.size() / 4) <= max_anchors) {
      pp.num_anchors = static_cast<int>(anchors.size() / 4);
      pp.len_per_anchor = 4;
      for (size_t i = 0; i < anchors.size(); ++i) {
        pp.anchors[i] = static_cast<float>(anchors[i]);
      }
    } else {
      ROS_WARN_STREAM("~anchors must hold a multiple of 4 values for at most "
                      << max_anchors << " anchors; ignoring and using the compiled default.");
    }
  }

  std::vector<double> bottom_heights;
  if (pnh.getParam("anchor_bottom_heights", bottom_heights) && !bottom_heights.empty()) {
    if (bottom_heights.size() <= 3) {
      pp.num_classes = static_cast<int>(bottom_heights.size());
      pp.anchor_bottom_heights = nvtype::Float3(
          static_cast<float>(bottom_heights[0]),
          bottom_heights.size() > 1 ? static_cast<float>(bottom_heights[1]) : 0.0f,
          bottom_heights.size() > 2 ? static_cast<float>(bottom_heights[2]) : 0.0f);
    } else {
      ROS_WARN_STREAM("~anchor_bottom_heights supports at most 3 classes; "
                      "ignoring and using the compiled default.");
    }
  }

  double dir_offset = pp.dir_offset;
  if (pnh.getParam("dir_offset", dir_offset)) {
    pp.dir_offset = static_cast<float>(dir_offset);
  }

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
    pnh_.param<std::string>("latency_topic", latency_topic_, "/pointpillar/inference_latency_ms");
    pnh_.param<std::string>("detections_topic", detections_topic_, "/pointpillar/detections");
    pnh_.param<bool>("enable_timer", enable_timer_, false);
    pnh_.param<int>("queue_size", queue_size_, 1);
    pnh_.param<double>("fps_log_interval", fps_log_interval_sec_, 5.0);
    pnh_.param<double>("score_thresh", score_thresh_, -1.0);
    pnh_.param<double>("nms_thresh", nms_thresh_, -1.0);

    if (!pnh_.getParam("class_names", class_names_) || class_names_.empty()) {
      class_names_ = {"Car", "Pedestrian", "Cyclist"};
    }

    auto vp = makeVoxelizationParam(pnh_);
    auto pp = makePostParam(vp, pnh_);
    // Negative (default) means "keep the compiled-in default from
    // PostProcessParameter"; set score_thresh/nms_thresh params to override
    // for a custom-trained model without recompiling.
    if (score_thresh_ >= 0.0) {
      pp.score_thresh = static_cast<float>(score_thresh_);
    }
    if (nms_thresh_ >= 0.0) {
      pp.nms_thresh = static_cast<float>(nms_thresh_);
    }

    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp;
    param.lidar_model = model_file_;
    param.lidar_post = pp;

    core_ = pointpillar::lidar::create_core(param);
    if (!core_) {
      throw std::runtime_error("Failed to create PointPillars core. Check model_file.");
    }

    core_->set_timer(enable_timer_);
    core_->print();
    checkRuntime(cudaStreamCreate(&stream_));

    marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 1);
    latency_pub_ = nh_.advertise<std_msgs::Float32>(latency_topic_, 10);
    detections_pub_ =
        nh_.advertise<pointpillar::Detection3DArray>(detections_topic_, 1);
    points_sub_ = nh_.subscribe(points_topic_, queue_size_,
                                &PointPillarRosNode::pointsCallback, this);

    stats_window_start_ = ros::Time::now();
    ROS_INFO_STREAM("PointPillars ROS node ready."
                    << " model_file=" << model_file_
                    << " points_topic=" << points_topic_
                    << " marker_topic=" << marker_topic_
                    << " latency_topic=" << latency_topic_
                    << " detections_topic=" << detections_topic_);
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
    const ros::WallTime inference_start = ros::WallTime::now();

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
      publishDetections({}, msg->header);
      return;
    }

    auto boxes = core_->forward(points.data(), static_cast<int>(points.size() / 4), stream_);
    const double latency_ms =
        (ros::WallTime::now() - inference_start).toSec() * 1000.0;
    publishLatency(latency_ms);
    updateStats(latency_ms, boxes.size(), msg->header.frame_id);
    publishBoxes(boxes, msg->header);
    publishDetections(boxes, msg->header);
  }

  void publishLatency(double latency_ms) {
    std_msgs::Float32 latency_msg;
    latency_msg.data = static_cast<float>(latency_ms);
    latency_pub_.publish(latency_msg);
  }

  void updateStats(double latency_ms, size_t box_count,
                   const std::string& frame_id) {
    total_frames_++;
    window_frames_++;
    total_latency_ms_ += latency_ms;
    window_latency_ms_ += latency_ms;

    const ros::Time now = ros::Time::now();
    if (stats_window_start_.isZero()) {
      stats_window_start_ = now;
      return;
    }

    const double window_sec = (now - stats_window_start_).toSec();
    if (window_sec < fps_log_interval_sec_) {
      return;
    }

    const double fps = window_sec > 0.0
                           ? static_cast<double>(window_frames_) / window_sec
                           : 0.0;
    const double avg_latency_ms =
        window_frames_ > 0 ? window_latency_ms_ / window_frames_ : 0.0;
    const double avg_total_latency_ms =
        total_frames_ > 0 ? total_latency_ms_ / total_frames_ : 0.0;

    ROS_INFO_STREAM("PointPillars stats: fps=" << fps
                    << " avg_latency_ms=" << avg_latency_ms
                    << " total_avg_latency_ms=" << avg_total_latency_ms
                    << " last_boxes=" << box_count
                    << " frame_id=" << frame_id);

    stats_window_start_ = now;
    window_frames_ = 0;
    window_latency_ms_ = 0.0;
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
    int label_id = 0;
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

      const float distance = std::sqrt(box.x * box.x + box.y * box.y + box.z * box.z);
      char label[32];
      std::snprintf(label, sizeof(label), "%.1fm", distance);

      visualization_msgs::Marker text_marker;
      text_marker.header = header;
      text_marker.ns = "pointpillar_distance";
      text_marker.id = label_id++;
      text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::Marker::ADD;
      text_marker.pose.position.x = box.x;
      text_marker.pose.position.y = box.y;
      text_marker.pose.position.z = box.z + std::max(box.h, 0.01f) / 2.0f + 0.3f;
      text_marker.pose.orientation.w = 1.0;
      text_marker.scale.z = 0.4;
      text_marker.color.r = 1.0f;
      text_marker.color.g = 1.0f;
      text_marker.color.b = 1.0f;
      text_marker.color.a = 1.0f;
      text_marker.text = label;
      text_marker.lifetime = ros::Duration(0.1);
      array.markers.push_back(text_marker);
    }

    marker_pub_.publish(array);
  }

  void publishDetections(const std::vector<pointpillar::lidar::BoundingBox>& boxes,
                         const std_msgs::Header& header) {
    pointpillar::Detection3DArray array;
    array.header = header;
    array.detections.reserve(boxes.size());

    for (const auto& box : boxes) {
      pointpillar::Detection3D det;
      det.header = header;
      det.label_id = box.id;
      det.label_name = (box.id >= 0 && static_cast<size_t>(box.id) < class_names_.size())
                            ? class_names_[box.id]
                            : "unknown";
      det.score = box.score;
      det.x = box.x;
      det.y = box.y;
      det.z = box.z;
      det.length = box.l;
      det.width = box.w;
      det.height = box.h;
      det.yaw = box.rt;
      det.distance = std::sqrt(box.x * box.x + box.y * box.y + box.z * box.z);
      array.detections.push_back(std::move(det));
    }

    detections_pub_.publish(array);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber points_sub_;
  ros::Publisher marker_pub_;
  ros::Publisher latency_pub_;
  ros::Publisher detections_pub_;

  std::shared_ptr<pointpillar::lidar::Core> core_;
  cudaStream_t stream_ = nullptr;
  std::mutex mutex_;

  std::string model_file_;
  std::string points_topic_;
  std::string marker_topic_;
  std::string latency_topic_;
  std::string detections_topic_;
  std::vector<std::string> class_names_;
  bool enable_timer_ = false;
  int queue_size_ = 1;
  double fps_log_interval_sec_ = 5.0;
  double score_thresh_ = -1.0;
  double nms_thresh_ = -1.0;
  ros::Time stats_window_start_;
  uint64_t total_frames_ = 0;
  uint64_t window_frames_ = 0;
  double total_latency_ms_ = 0.0;
  double window_latency_ms_ = 0.0;
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
