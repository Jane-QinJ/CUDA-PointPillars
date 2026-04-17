#!/usr/bin/env python3

import argparse
import glob
import os
import struct

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2


FIELDS = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
]


def load_bin_points(path):
    with open(path, "rb") as f:
        payload = f.read()

    points = [p for p in struct.iter_unpack("ffff", payload)]
    if not points:
        raise RuntimeError("No points found in {}".format(path))
    return points


def collect_pointclouds(data_path):
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, "*.bin")))
    else:
        files = [data_path]

    if not files:
        raise RuntimeError("No .bin files found in {}".format(data_path))

    clouds = []
    for path in files:
        clouds.append((path, load_bin_points(path)))
    return clouds


def main():
    parser = argparse.ArgumentParser(
        description="Publish KITTI-style point clouds from .bin files to ROS."
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Path to a .bin file or a directory of .bin files",
    )
    parser.add_argument(
        "--topic",
        default="/kitti/velo/pointcloud",
        help="PointCloud2 topic to publish",
    )
    parser.add_argument(
        "--frame-id",
        default="velo_link",
        help="frame_id to stamp into PointCloud2",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Publish rate in Hz",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop over the dataset continuously",
    )
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("kitti_pointcloud_publisher", anonymous=False)
    pub = rospy.Publisher(args.topic, PointCloud2, queue_size=1)

    clouds = collect_pointclouds(args.data)
    rospy.loginfo(
        "Loaded %d point clouds from %s, publishing on %s at %.2f Hz",
        len(clouds),
        args.data,
        args.topic,
        args.rate,
    )

    rate = rospy.Rate(args.rate)

    while not rospy.is_shutdown():
        for path, points in clouds:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = args.frame_id
            msg = pc2.create_cloud(header, FIELDS, points)
            pub.publish(msg)
            rospy.logdebug("Published %s with %d points", path, len(points))
            rate.sleep()

        if not args.loop:
            break


if __name__ == "__main__":
    main()
