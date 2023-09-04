#!/usr/bin/env python3


import tf.transformations as tft

roll = 0.0194281
pitch = 0.0127725
yaw = 3.12124

quaternion = tft.quaternion_from_euler(roll, pitch, yaw)
print("Quaternion:", quaternion)
