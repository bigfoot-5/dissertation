#!/usr/bin/env python3

# Import Mask R-CNN model and other necessary libraries here
import numpy as np
import argparse
import cv2
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as T
import torch
from scripts.utils3 import position2pose
# from mask_rcnn import MaskRCNN, COCO_INSTANCE_CATEGORY_NAMES, COLORS
import time
from stream import Stream
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
import pyrealsense2 as rs
from cv_bridge import CvBridge
# Define the callback function to process the image

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


class MaskRCNN:
    def __init__(self,
                 threshold: float = 0.92):
        # Set threshold score and detection target
        self.threshold = threshold

        # Initialize detection network
        self.model = maskrcnn_resnet50_fpn(
            weights="MaskRCNN_ResNet50_FPN_Weights.COCO_V1", progress=True, num_classes=91)
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

        return masks, boxes, labels

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
            # apply a matching colour to each mask
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

def image_callback(msg):
    bridge = CvBridge()
    try:
        # Convert the ROS Image message to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Perform segmentation using Mask R-CNN
        # Preprocess the image
        transform = transforms.Compose([transforms.ToTensor()])
        input_image = transform(cv_image).unsqueeze(0)

        # Run inference using your Mask R-CNN model
        rcnn = MaskRCNN()  # Initialize your Mask R-CNN model
        output = rcnn(input_image)

        # Get the segmentation mask and apply it to the original image
        mask = output['masks'][0, 0].detach().numpy()
        segmented_image = cv_image.copy()
        segmented_image[mask == 0] = [0, 0, 0]  # Set non-segmented regions to black

        # Publish the segmented image on a new ROS topic
        segmented_msg = bridge.cv2_to_imgmsg(segmented_image, "bgr8")
        segmentation_pub.publish(segmented_msg)

        # Convert the original image to a ROS Image message
        original_msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        original_pub.publish(original_msg)

    except Exception as e:
        print(e)

def main():
    rospy.init_node('maskrcnn_segmentation_node', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    global segmentation_pub, original_pub
    segmentation_pub = rospy.Publisher('/camera/color/segmented', Image, queue_size=10)
    original_pub = rospy.Publisher('/camera/color/original', Image, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    main()
