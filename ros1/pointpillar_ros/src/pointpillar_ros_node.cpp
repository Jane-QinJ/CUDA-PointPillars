#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>

#include "pointpillar.hpp"

namespace {

std::vector<double> loadDoubleArray(ros::NodeHandle& nh, const std::string& key, std::size_t expected_size) {
    std::vector<double> values;
    if (!nh.getParam(key, values)) {
        throw std::runtime_error("Missing ROS param: " + key);
    }
    if (values.size() != expected_size) {
        throw std::runtime_error("ROS param " + key + " size mismatch");
    }
    return values;
}

std::vector<std::string> loadStringArray(ros::NodeHandle& nh, const std::string& key) {
    std::vector<std::string> values;
    nh.getParam(key, values);
    return values;
}

pointpillar::lidar::CoreParameter loadCoreParameter(ros::NodeHandle& pnh) {
    pointpillar::lidar::CoreParameter param;

    std::string engine_path;
    if (!pnh.getParam("engine_path", engine_path) || engine_path.empty()) {
        throw std::runtime_error("ROS param engine_path is required");
    }

    const auto point_cloud_range = loadDoubleArray(pnh, "point_cloud_range", 6);
    const auto voxel_size = loadDoubleArray(pnh, "voxel_size", 3);
    const auto anchors = loadDoubleArray(pnh, "anchors", 8);
    const auto anchor_bottom_heights = loadDoubleArray(pnh, "anchor_bottom_heights", 3);

    // These runtime parameters must stay consistent with the model export config.
    auto& vp = param.voxelization;
    vp.min_range = nvtype::Float3(point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]);
    vp.max_range = nvtype::Float3(point_cloud_range[3], point_cloud_range[4], point_cloud_range[5]);
    vp.voxel_size = nvtype::Float3(voxel_size[0], voxel_size[1], voxel_size[2]);
    vp.grid_size = vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
    pnh.param("max_voxels", vp.max_voxels, 40000);
    pnh.param("max_points_per_voxel", vp.max_points_per_voxel, 32);
    pnh.param("max_points", vp.max_points, 300000);
    pnh.param("num_feature", vp.num_feature, 4);

    // These postprocess values must match the exported dense head layout.
    auto& pp = param.lidar_post;
    pp.min_range = vp.min_range;
    pp.max_range = vp.max_range;
    pp.feature_size = nvtype::Int2(vp.grid_size.x / 2, vp.grid_size.y / 2);
    pnh.param("num_classes", pp.num_classes, 1);
    pnh.param("num_anchors", pp.num_anchors, 2);
    pnh.param("score_thresh", pp.score_thresh, 0.1f);
    pnh.param("nms_thresh", pp.nms_thresh, 0.01f);
    pnh.param("dir_offset", pp.dir_offset, 0.78539f);

    for (std::size_t i = 0; i < anchors.size(); ++i) {
        pp.anchors[i] = static_cast<float>(anchors[i]);
    }
    pp.anchor_bottom_heights = nvtype::Float3(
        anchor_bottom_heights[0], anchor_bottom_heights[1], anchor_bottom_heights[2]
    );

    param.lidar_model = engine_path;
    return param;
}

bool hasField(const sensor_msgs::PointCloud2& msg, const std::string& field_name) {
    for (const auto& field : msg.fields) {
        if (field.name == field_name) {
            return true;
        }
    }
    return false;
}

std_msgs::ColorRGBA colorForClass(int class_id) {
    std_msgs::ColorRGBA color;
    color.a = 1.0;
    switch (class_id % 3) {
        case 0:
            color.r = 0.95;
            color.g = 0.25;
            color.b = 0.25;
            break;
        case 1:
            color.r = 0.15;
            color.g = 0.75;
            color.b = 0.25;
            break;
        default:
            color.r = 0.15;
            color.g = 0.45;
            color.b = 0.95;
            break;
    }
    return color;
}

class PointPillarsRosNode {
  public:
    PointPillarsRosNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh) {
        pointcloud_topic_ = pnh_.param<std::string>("pointcloud_topic", "/points_raw");
        marker_topic_ = pnh_.param<std::string>("marker_topic", "/pointpillar/detections");
        marker_frame_id_ = pnh_.param<std::string>("marker_frame_id", "");
        use_input_frame_ = pnh_.param("use_input_frame", true);
        marker_lifetime_sec_ = pnh_.param("marker_lifetime", 0.1);
        line_width_ = pnh_.param("line_width", 0.08);
        min_confidence_ = pnh_.param("min_confidence", 0.5);
        enable_timer_ = pnh_.param("enable_timer", false);
        print_fps_ = pnh_.param("print_fps", true);
        fps_window_size_ = pnh_.param("fps_window_size", 30);
        publish_filtered_pointcloud_ = pnh_.param("publish_filtered_pointcloud", true);
        filtered_pointcloud_topic_ = pnh_.param<std::string>("filtered_pointcloud_topic", "/pointpillar/filtered_points");
        class_names_ = loadStringArray(pnh_, "class_names");
        if (class_names_.empty()) {
            class_names_.push_back("Pedestrian");
        }

        core_param_ = loadCoreParameter(pnh_);
        core_ = pointpillar::lidar::create_core(core_param_);
        if (!core_) {
            throw std::runtime_error("Failed to create PointPillars core");
        }
        core_->set_timer(enable_timer_);
        core_->print();

        cudaError_t err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreate failed");
        }

        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 1);
        if (publish_filtered_pointcloud_) {
            filtered_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(filtered_pointcloud_topic_, 1);
        }
        point_sub_ = nh_.subscribe(pointcloud_topic_, 1, &PointPillarsRosNode::pointCloudCallback, this);
    }

    ~PointPillarsRosNode() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

  private:
    bool isPointInRange(float x, float y, float z) const {
        const auto& min_range = core_param_.voxelization.min_range;
        const auto& max_range = core_param_.voxelization.max_range;
        return x >= min_range.x && x < max_range.x &&
               y >= min_range.y && y < max_range.y &&
               z >= min_range.z && z < max_range.z;
    }

    void publishFilteredCloud(
        const std_msgs::Header& header,
        const std::vector<float>& points,
        bool has_intensity) {
        // Publish only the points that are actually sent into the detector.
        if (!publish_filtered_pointcloud_) {
            return;
        }

        sensor_msgs::PointCloud2 cloud;
        cloud.header = header;
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        if (has_intensity) {
            modifier.setPointCloud2Fields(
                4,
                "x", 1, sensor_msgs::PointField::FLOAT32,
                "y", 1, sensor_msgs::PointField::FLOAT32,
                "z", 1, sensor_msgs::PointField::FLOAT32,
                "intensity", 1, sensor_msgs::PointField::FLOAT32
            );
        } else {
            modifier.setPointCloud2Fields(
                3,
                "x", 1, sensor_msgs::PointField::FLOAT32,
                "y", 1, sensor_msgs::PointField::FLOAT32,
                "z", 1, sensor_msgs::PointField::FLOAT32
            );
        }

        const std::size_t point_count = points.size() / 4;
        modifier.resize(point_count);

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

        if (has_intensity) {
            sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloud, "intensity");
            for (std::size_t i = 0; i < point_count; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
                const std::size_t base = i * 4;
                *iter_x = points[base];
                *iter_y = points[base + 1];
                *iter_z = points[base + 2];
                *iter_intensity = points[base + 3];
            }
        } else {
            for (std::size_t i = 0; i < point_count; ++i, ++iter_x, ++iter_y, ++iter_z) {
                const std::size_t base = i * 4;
                *iter_x = points[base];
                *iter_y = points[base + 1];
                *iter_z = points[base + 2];
            }
        }

        filtered_cloud_pub_.publish(cloud);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        if (msg->width == 0 || msg->height == 0) {
            return;
        }

        std::vector<float> points;
        points.reserve(static_cast<std::size_t>(msg->width) * msg->height * 4);
        const bool has_intensity = hasField(*msg, "intensity");

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        if (has_intensity) {
            sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(*msg, "intensity");
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
                if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) {
                    continue;
                }
                if (!isPointInRange(*iter_x, *iter_y, *iter_z)) {
                    continue;
                }
                points.push_back(*iter_x);
                points.push_back(*iter_y);
                points.push_back(*iter_z);
                points.push_back(std::isfinite(*iter_intensity) ? *iter_intensity : 0.0f);
            }
        } else {
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
                if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) {
                    continue;
                }
                if (!isPointInRange(*iter_x, *iter_y, *iter_z)) {
                    continue;
                }
                points.push_back(*iter_x);
                points.push_back(*iter_y);
                points.push_back(*iter_z);
                points.push_back(0.0f);
            }
        }

        if (points.empty()) {
            return;
        }

        const ros::WallTime start_time = ros::WallTime::now();
        publishFilteredCloud(msg->header, points, has_intensity);
        // Run TensorRT inference on points already cropped by point_cloud_range.
        auto boxes = core_->forward(points.data(), static_cast<int>(points.size() / 4), stream_);
        updateFps((ros::WallTime::now() - start_time).toSec());
        marker_pub_.publish(buildMarkers(msg->header, boxes));
    }

    void updateFps(double frame_seconds) {
        if (!print_fps_ || frame_seconds <= 0.0) {
            return;
        }

        fps_window_.push_back(frame_seconds);
        if (fps_window_.size() > fps_window_size_) {
            fps_window_.erase(fps_window_.begin());
        }

        const double total = std::accumulate(fps_window_.begin(), fps_window_.end(), 0.0);
        if (total <= 0.0) {
            return;
        }

        const double avg_frame_ms = (total / static_cast<double>(fps_window_.size())) * 1000.0;
        const double avg_fps = static_cast<double>(fps_window_.size()) / total;
        ROS_INFO_THROTTLE(1.0, "PointPillars avg %.2f FPS (%.2f ms, window=%zu)", avg_fps, avg_frame_ms, fps_window_.size());
    }

    std::string classNameForId(int class_id) const {
        if (class_id >= 0 && static_cast<std::size_t>(class_id) < class_names_.size()) {
            return class_names_[class_id];
        }
        return "Class" + std::to_string(class_id);
    }

    std::string labelText(const pointpillar::lidar::BoundingBox& box) const {
        const double distance = std::sqrt(
            static_cast<double>(box.x) * box.x +
            static_cast<double>(box.y) * box.y +
            static_cast<double>(box.z) * box.z
        );
        std::ostringstream oss;
        oss << classNameForId(box.id) << " " << std::fixed << std::setprecision(2) << distance << "m";
        return oss.str();
    }

    visualization_msgs::MarkerArray buildMarkers(
        const std_msgs::Header& header,
        const std::vector<pointpillar::lidar::BoundingBox>& boxes) const {
        visualization_msgs::MarkerArray array;

        visualization_msgs::Marker clear;
        clear.header = header;
        clear.ns = "pointpillar";
        clear.action = visualization_msgs::Marker::DELETEALL;
        array.markers.push_back(clear);

        const std::string frame_id = use_input_frame_ ? header.frame_id : marker_frame_id_;
        ros::Duration lifetime(marker_lifetime_sec_);
        int marker_index = 0;

        for (const auto& box : boxes) {
            if (box.score < min_confidence_) {
                continue;
            }
            visualization_msgs::Marker marker;
            marker.header.stamp = header.stamp;
            marker.header.frame_id = frame_id;
            marker.ns = "pointpillar_boxes";
            marker.id = marker_index;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = box.x;
            marker.pose.position.y = box.y;
            marker.pose.position.z = box.z;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = std::sin(box.rt * 0.5f);
            marker.pose.orientation.w = std::cos(box.rt * 0.5f);
            marker.scale.x = box.l;
            marker.scale.y = box.w;
            marker.scale.z = box.h;
            marker.color = colorForClass(box.id);
            marker.color.a = std::max(0.2f, std::min(1.0f, box.score));
            marker.lifetime = lifetime;
            array.markers.push_back(marker);

            visualization_msgs::Marker text;
            text.header = marker.header;
            text.ns = "pointpillar_scores";
            text.id = marker_index + 100000;
            text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::Marker::ADD;
            text.pose.position.x = box.x;
            text.pose.position.y = box.y;
            text.pose.position.z = box.z + box.h * 0.7f;
            text.scale.z = std::max(0.6, line_width_ * 8.0);
            text.color.r = 1.0;
            text.color.g = 1.0;
            text.color.b = 1.0;
            text.color.a = 1.0;
            text.lifetime = lifetime;
            text.text = labelText(box);
            array.markers.push_back(text);
            ++marker_index;
        }

        return array;
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber point_sub_;
    ros::Publisher marker_pub_;
    ros::Publisher filtered_cloud_pub_;
    std::shared_ptr<pointpillar::lidar::Core> core_;
    pointpillar::lidar::CoreParameter core_param_;
    cudaStream_t stream_ = nullptr;

    std::string pointcloud_topic_;
    std::string marker_topic_;
    std::string marker_frame_id_;
    std::string filtered_pointcloud_topic_;
    std::vector<std::string> class_names_;
    bool use_input_frame_ = true;
    bool enable_timer_ = false;
    bool print_fps_ = true;
    bool publish_filtered_pointcloud_ = true;
    int fps_window_size_ = 30;
    double marker_lifetime_sec_ = 0.1;
    double line_width_ = 0.08;
    double min_confidence_ = 0.5;
    std::vector<double> fps_window_;
};

}  // namespace

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointpillar_ros");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    try {
        PointPillarsRosNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("%s", e.what());
        return 1;
    }

    return 0;
}
