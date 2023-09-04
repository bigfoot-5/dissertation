#!/usr/bin/env python3

from utils import position2pose, project_point, img_to_cv2
from mask_rcnn import MaskRCNN
import time
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
import rospy
from cv_bridge import CvBridge


bridge = CvBridge()


if __name__ == '__main__':
    mask_rcnn = MaskRCNN()
    segment_pub = rospy.Publisher(
        '/mask_rcnn/segment_image', Image, queue_size=1)
    # segment_info_pub = rospy.Publisher(
    #     '/mask_rcnn/camera_info', CameraInfo, queue_size=1)

    def callback(rgb_image):
        print("entered")
        rgb_image = img_to_cv2(rgb_image)
        masks, boxes, labels = mask_rcnn.forward(rgb_image)
        print(labels)
        image = mask_rcnn.get_segmentation_image(rgb_image, masks, boxes, labels)
        target= input()
        target_centroid = mask_rcnn.get_target_pixel(boxes, labels, target)
        # segment_info_pub.publish(rgb_info)
        segment_pub.publish(bridge.cv2_to_imgmsg(image, 'bgr8'))

    try:
        rospy.init_node('mask_rcnn_node', anonymous=True)
        rgb_image = message_filters.Subscriber(
            "/rrbot/camera1/image_raw", Image)
        # rgb_info = message_filters.Subscriber(
        #     "/rrbot/camera1/camera_info", CameraInfo)
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_image],10, 0.05, queue_size=100, allow_headerless=True)
        ts.registerCallback(callback)
        time.sleep(3)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    except Exception as e:
        print(e)
