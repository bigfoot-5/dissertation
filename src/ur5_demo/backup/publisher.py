import time
from stream import Stream
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
import pyrealsense2 as rs
import argparse
from cv_bridge import CvBridge
import cv2
import random
from torchvision.models.detection import maskrcnn_resnet50_fpn
import pyrealsense2 as rs2
import torch
from torchvision import transforms as T
import tf

bridge = CvBridge()


# Classes names from coco
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Make a different colour for each of the object classes
COLORS = np.random.uniform(
    0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))


def position2pose(position: list = [0, 0, 0],
                  frame: str = "camera_color_optical_frame") -> PoseStamped:
    target_pose = PoseStamped()
    target_pose.header.frame_id = frame
    target_pose.pose.position.x = position[0]
    target_pose.pose.position.y = position[1]
    target_pose.pose.position.z = position[2]

    print(target_pose.pose.position)
    return target_pose


class MaskRCNN:
    def __init__(self,
                 threshold: float = 0.92):
        # Set threshold score and detection target
        self.threshold = threshold

        # Initialize detection network
        self.model = maskrcnn_resnet50_fpn(
            pretrained=True, progress=True, num_classes=91)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
        self.model.to(self.device)

        # Convert from numpy array (H x W x C) in the range [0, 255]
        #   to tensor (C x H x W) in the range [0.0, 1.0]
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def forward(self,
                image: np.ndarray,
                print_results: bool = False):

        # Transform the image
        image = self.transform(image)
        # Add a batch dimension
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Forward pass of the image through the model
            outputs = self.model(image)[0]

        # Get all the scores
        scores = list(outputs['scores'].detach().cpu().numpy())
        # Index of those scores which are above a certain threshold
        thresholded_preds_inidices = [
            scores.index(i) for i in scores if i > self.threshold]

        thresholded_preds_count = len(thresholded_preds_inidices)

        # Get the masks
        masks = (outputs['masks'] > 0.5).squeeze().detach().cpu().numpy()
        # Discard masks for objects which are below threshold
        masks = masks[:thresholded_preds_count]
        # Get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                 for i in outputs['boxes'].detach().cpu()]
        # Discard bounding boxes below threshold value
        boxes = boxes[:thresholded_preds_count]

        # get labels that pass the treshold and are in the list of classes
        labels = outputs['labels'][:thresholded_preds_count]
        colours = [COLORS[i] for i in labels]

        # Get the classes labels
        labels = [COCO_INSTANCE_CATEGORY_NAMES[i]
                  for i in labels]
        if print_results:
            # Print results as 'Label: Score'
            scores = scores[:thresholded_preds_count]
            results = ''
            for i in range(thresholded_preds_count):
                results += f'{labels[i]}: {scores[i]} '
            print(results)

        return masks, boxes, labels, colours

    def get_segmentation_image(self,
                               image: np.array,
                               masks:  list,
                               boxes: list,
                               labels: list) -> np.array:
        alpha = 1
        beta = 0.6  # transparency for the segmentation map
        gamma = 0  # scalar added to each sum
        for i in range(len(masks)):
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            # apply a randon color mask to each object
            color = COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(labels[i])]
            red_map[masks[i] == 1], green_map[masks[i]
                                              == 1], blue_map[masks[i] == 1] = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            # convert the original PIL image into NumPy format
            image = np.array(image)
            # convert from RGN to OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            # draw the bounding boxes around the objects
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                          thickness=2)
            # put the label text above the objects
            cv2.putText(image, labels[i], (boxes[i][0][0], boxes[i][0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        thickness=2, lineType=cv2.LINE_AA)
        return image

    def get_target_pixel(self,
                         boxes: list,
                         labels: list,
                         target_class: str = 'keyboard') -> tuple:
        # Drawing bounding boxes
        print("Found {} objects".format(len(boxes)), labels)

        try:
            # Get the index of the target class
            target_class_index = labels.index(target_class)

            target_box = boxes[target_class_index]

            # Detect target
            # Calculate 2D position of target centroid
            [x1, y1], [x2, y2] = target_box
            x1, x2 = max(x1, 0), min(x2, 639)
            y1, y2 = max(y1, 0), min(y2, 479)

            target_centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            print(f'Found {target_class} at index {target_class_index}')
            print(f'Target centroid: {target_centroid}')

            return target_centroid
        except IndexError as e:
            print(e)
            return None
        except ValueError as e:
            print(e)
            return None

    # converts 3D position to PoseStamped message
    @staticmethod
    def quaternion_from_euler(roll, pitch, yaw):
        q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        return q

    def xyz_to_pose(self,
                    xyz: list,
                    frame: str = "camera_color_optical_frame") -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.pose.position.x = xyz[0]
        pose.pose.position.y = xyz[1]
        pose.pose.position.z = xyz[2]

        orientation = self.quaternion_from_euler(0, 0, 0)
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]

        return pose

    # projects a point given the depth, 2D position and camera parameters shift-F6
    def project_point(self,
                      depth_image: np.ndarray,
                      xy: list,
                      camera_info: CameraInfo) -> list:
        depth = depth_image[xy[0], xy[1]] / 1000
        # convert 2D position to 3D position
        xyz = [depth * (xy[0] - camera_info.K[2]) / camera_info.K[0],
               depth * (xy[1] - camera_info.K[5]) / camera_info.K[5],
               depth]

        return xyz

    @staticmethod
    def make_intrinsic(camera_info: CameraInfo) -> rs2.intrinsics:
        intrinsics = rs2.intrinsics()
        intrinsics.width = camera_info.width
        intrinsics.height = camera_info.height
        intrinsics.ppx = camera_info.K[2]
        intrinsics.ppy = camera_info.K[5]
        intrinsics.fx = camera_info.K[0]
        intrinsics.fy = camera_info.K[4]
        intrinsics.coeffs = [i for i in camera_info.D]
        return intrinsics


if __name__ == '__main__':

    rospy.init_node('vision', anonymous=True)
    stream = Stream()
    stream.start()
    rcnn = MaskRCNN()
    pose_pub = rospy.Publisher('/vision/pose',
                               PoseStamped, queue_size=10)
    segmentation_pub = rospy.Publisher('/vision/segmentation',
                                       Image, queue_size=10)
    try:

        while not rospy.is_shutdown():

            color_image, depth_image = stream.get_images()

            # get the masks, bounding boxes, and labels from the RCNN
            masks, bounding_boxes, labels, colours = rcnn.forward(color_image)

            # get the segmentation Image
            segmentation_image = rcnn.get_segmentation_image(
                color_image, masks, bounding_boxes, labels)

            # get the target pixel
            target_centroid = rcnn.get_target_pixel(bounding_boxes, labels,
                                                    'mouse')

            if target_centroid is not None:
                y, x = target_centroid
                z = depth_image[int(x), int(y)] / 1000

                # 2d position to 3d position
                position3D = rs.rs2_deproject_pixel_to_point(
                    stream.intrinsics, [x, y], z)

                print(f'Target at {position3D}')
                pose = rcnn.xyz_to_pose(position3D)

                rospy.loginfo(pose)
                pose_pub.publish(pose)
                print("published pose", pose)

            segmentation_pub.publish(
                bridge.cv2_to_imgmsg(segmentation_image, 'bgr8'))
            time.sleep(4)

    except rospy.ROSInterruptException:
        stream.stop()
        pass
    except KeyboardInterrupt:
        stream.stop()
        pass
