#!/usr/bin/env python3

import rclpy
import numpy as np
import cv2
import rosbag2_py
import os
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import QuaternionStamped, Vector3Stamped
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, PoseStamped
from kitti2rosbag2.utils.kitti_utils import KITTIOdometryDataset
from kitti2rosbag2.utils.quaternion import Quaternion
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rclpy.serialization import serialize_message

class Kitti_Odom(Node):
    def __init__(self):
        super().__init__("kitti_rec")

        # parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('sequence', rclpy.Parameter.Type.INTEGER),
                ('data_dir', rclpy.Parameter.Type.STRING),
                ('odom', rclpy.Parameter.Type.BOOL),
                ('odom_dir', rclpy.Parameter.Type.STRING),
                ('bag_dir', rclpy.Parameter.Type.STRING),
            ]
        )

        sequence = self.get_parameter('sequence').value
        data_dir = self.get_parameter('data_dir').get_parameter_value().string_value
        odom = self.get_parameter('odom').value
        bag_dir = self.get_parameter('bag_dir').get_parameter_value().string_value
        if odom == True:
            odom_dir = self.get_parameter('odom_dir').get_parameter_value().string_value
        else:
            odom_dir = None
        
        self.kitti_dataset = KITTIOdometryDataset(data_dir, sequence, odom_dir)
        self.bridge = CvBridge()
        self.counter = 0
        self.counter_limit = len(self.kitti_dataset.left_images()) - 1 
        
        self.left_imgs = self.kitti_dataset.left_images()
        self.right_imgs = self.kitti_dataset.right_images()
        self.times_file = self.kitti_dataset.times_file()
        self.odom = odom
        if odom == True:
            try:
                self.ground_truth = self.kitti_dataset.odom_pose()
            except FileNotFoundError as filenotfounderror:
                self.get_logger().error("Error: {}".format(filenotfounderror))
                rclpy.shutdown()
                return

        # rosbag writer
        self.writer = rosbag2_py.SequentialWriter()
        if os.path.exists(bag_dir):
            self.get_logger().info(f'The directory {bag_dir} already exists. Shutting down...')
            rclpy.shutdown()
        else:
            storage_options = rosbag2_py._storage.StorageOptions(uri=bag_dir, storage_id='sqlite3')
            converter_options = rosbag2_py._storage.ConverterOptions('', '')
            self.writer.open(storage_options, converter_options)

        left_img_topic_info = rosbag2_py._storage.TopicMetadata(id=0, name='/camera2/left/image_raw', type='sensor_msgs/msg/Image', serialization_format='cdr')
        right_img_topic_info = rosbag2_py._storage.TopicMetadata(id=1, name='/camera3/right/image_raw', type='sensor_msgs/msg/Image', serialization_format='cdr')
        odom_topic_info = rosbag2_py._storage.TopicMetadata(id=2, name='/car/base/odom', type='nav_msgs/msg/Odometry', serialization_format='cdr')
        path_topic_info = rosbag2_py._storage.TopicMetadata(id=3, name='/car/base/odom_path', type='nav_msgs/msg/Path', serialization_format='cdr')
        left_cam_topic_info = rosbag2_py._storage.TopicMetadata(id=4, name='/camera2/left/camera_info', type='sensor_msgs/msg/CameraInfo', serialization_format='cdr')
        right_cam_topic_info = rosbag2_py._storage.TopicMetadata(id=5, name='/camera3/right/camera_info', type='sensor_msgs/msg/CameraInfo', serialization_format='cdr')
        right_cam_topic_info = rosbag2_py._storage.TopicMetadata(id=5, name='/camera3/right/camera_info', type='sensor_msgs/msg/CameraInfo', serialization_format='cdr')
        pose_rotation_stamped = rosbag2_py._storage.TopicMetadata(id=6, name='/pose/base/rotation_stamped', type='geometry_msgs/msg/QuaternionStamped', serialization_format='cdr')
        pose_translation_stamped = rosbag2_py._storage.TopicMetadata(id=7, name='/pose/base/translation_stamped', type='geometry_msgs/msg/Vector3Stamped', serialization_format='cdr')

        self.writer.create_topic(left_img_topic_info)
        self.writer.create_topic(right_img_topic_info)
        self.writer.create_topic(odom_topic_info)
        self.writer.create_topic(path_topic_info)
        self.writer.create_topic(left_cam_topic_info)
        self.writer.create_topic(right_cam_topic_info)
        self.writer.create_topic(pose_rotation_stamped)
        self.writer.create_topic(pose_translation_stamped)

        # path msg
        self.p_msg = Path()

        # initialization
        self.timer = self.create_timer(0.05, self.rec_callback)


    def rec_callback(self):
        time = self.times_file[self.counter]
        timestamp_ns = int(time * 1e9) # nanoseconds

        # retrieving images and writing to bag
        left_image = cv2.imread(self.left_imgs[self.counter])
        right_image = cv2.imread(self.right_imgs[self.counter])
        left_img_msg = self.bridge.cv2_to_imgmsg(left_image, encoding='passthrough')
        self.writer.write('/camera2/left/image_raw', serialize_message(left_img_msg), timestamp_ns)
        right_img_msg = self.bridge.cv2_to_imgmsg(right_image, encoding='passthrough')
        self.writer.write('/camera3/right/image_raw', serialize_message(right_img_msg), timestamp_ns)

        # retrieving project mtx and writing to bag
        p_mtx2 = self.kitti_dataset.projection_matrix(1)
        self.rec_camera_info(p_mtx2, '/camera2/left/camera_info', timestamp_ns)
        p_mtx3 = self.kitti_dataset.projection_matrix(2)
        self.rec_camera_info(p_mtx3, '/camera3/right/camera_info', timestamp_ns)

        if self.odom == True:
            translation = self.ground_truth[self.counter][:3,3]
            quaternion = Quaternion()
            quaternion = quaternion.rotationmtx_to_quaternion(self.ground_truth[self.counter][:3, :3])
            self.rec_odom_msg(translation, quaternion, timestamp_ns)
            self.rec_odom_msg2(translation, quaternion, timestamp_ns)

        self.get_logger().info(f'{self.counter}-Images Processed, on timestamp_ns {timestamp_ns}')
        if self.counter >= self.counter_limit:
            self.get_logger().info('All images and poses published. Stopping...')
            rclpy.shutdown()
            self.timer.cancel()

        self.counter += 1
        return
    
    def rec_camera_info(self, mtx, topic, timestamp):
        camera_info_msg_2 = CameraInfo()
        camera_info_msg_2.p = mtx.flatten()
        self.writer.write(topic, serialize_message(camera_info_msg_2), timestamp)   
        return
    
    def rec_odom_msg(self, translation, quaternion, timestamp):
        odom_msg = Odometry()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "odom"
        odom_msg.pose.pose.position.z = -translation[1]
        odom_msg.pose.pose.position.y = -translation[2] 
        odom_msg.pose.pose.position.x = -translation[0] 
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]
        self.writer.write('/car/base/odom', serialize_message(odom_msg), timestamp)
        self.rec_path_msg(odom_msg, timestamp)
        return

    def rec_odom_msg2(self, translation, quaternion, timestamp_ns):
        rotation_stamped = QuaternionStamped()
        translation_stamped = Vector3Stamped()

        translation_stamped.header.frame_id = "map"
        translation_stamped.header.stamp  = self.ns_to_time(timestamp_ns)
        # self.get_logger().info(f'head-time secs {translation_stamped.header.stamp.sec} nsec {translation_stamped.header.stamp.nanosec}')
        translation_stamped.vector.z = -translation[1]
        translation_stamped.vector.y = -translation[2] 
        translation_stamped.vector.x = -translation[0]
        self.writer.write('/pose/base/translation_stamped', serialize_message(translation_stamped), timestamp_ns)

        rotation_stamped.header.frame_id = "map"
        rotation_stamped.header.stamp  = self.ns_to_time(timestamp_ns)
        rotation_stamped.quaternion.x = quaternion[0]
        rotation_stamped.quaternion.y = quaternion[1]
        rotation_stamped.quaternion.z = quaternion[2]
        rotation_stamped.quaternion.w = quaternion[3]
        self.writer.write('/pose/base/rotation_stamped', serialize_message(rotation_stamped), timestamp_ns)
        return
    
    def rec_path_msg(self, odom_msg, timestamp):
        pose= PoseStamped()
        pose.pose = odom_msg.pose.pose
        pose.header.frame_id = "odom"
        self.p_msg.poses.append(pose)
        self.p_msg.header.frame_id = "map"
        self.writer.write('/car/base/odom_path', serialize_message(self.p_msg), timestamp)
        return

    @staticmethod
    def ns_to_time(timestamp_ns):
        # Convert nanoseconds to a Time message
        time_msg = Time()
        time_msg.sec = timestamp_ns // 1_000_000_000  # Seconds part
        time_msg.nanosec = timestamp_ns % 1_000_000_000  # Nanoseconds part
        return time_msg

def main(args=None):
    rclpy.init(args=args)
    node = Kitti_Odom()
    rclpy.spin(node)
    try:
        rclpy.shutdown()
    except Exception as e:
        pass

if __name__ == '__main__':
    main()
