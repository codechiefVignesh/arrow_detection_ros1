#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

class RealtimeArrowDetectionNode:
    def __init__(self):
        rospy.init_node('realtime_arrow_detection_node', anonymous=True)
        self.model = self.load_saved_model('/home/vignesh/Documents/svm_mars/file/kaggle/working/efficient_model')  # Update with your model path
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

    def array_to_direction(self, array):
        # Your existing array_to_direction function
        if array[0][0] > array[0][1] and array[0][0] > array[0][2]:
            rospy.loginfo("left")
        elif array[0][1] > array[0][0] and array[0][1] > array[0][2]:
            rospy.loginfo("right")
        elif array[0][2] > array[0][1] and array[0][2] > array[0][0]:
            rospy.loginfo("up")
        else:
            rospy.loginfo("down")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            cv_image = cv2.resize(cv_image, (300, 300))
            img = np.asarray(cv_image)
            img = np.expand_dims(img, axis=0)
            output = self.model.predict(img)
            self.array_to_direction(output)
            rospy.loginfo("Processed image and predicted direction.")
        except CvBridgeError as e:
            rospy.logerr(e)

    def load_saved_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        return model

if __name__ == '__main__':
    try:
        node = RealtimeArrowDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
