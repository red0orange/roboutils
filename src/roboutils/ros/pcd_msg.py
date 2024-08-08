import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


def array_to_pointcloud2(cloud_array, frame_id="base_link"):
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # 定义 PointCloud2 字段
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # 根据需要添加其他字段，例如强度 (intensity)
    ]

    # 将 numpy 数组转换为 PointCloud2 消息
    cloud_msg = pc2.create_cloud(header, fields, cloud_array)
    return cloud_msg


def pointcloud2_to_array(cloud_msg):
    # 从 PointCloud2 消息中提取字段信息
    field_names = [field.name for field in cloud_msg.fields]
    # 生成点云的 numpy 数组
    cloud_array = np.array(list(pc2.read_points(cloud_msg, skip_nans=True, field_names=field_names)))
    return cloud_array