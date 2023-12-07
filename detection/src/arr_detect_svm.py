#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import joblib

class ArrowDetectorROS:
    def __init__(self, svm_model_pathh):
        rospy.init_node('arrow_detector_svm', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        try:
            self.svm_model = joblib.load(svm_model_pathh)
        except Exception as e:
            print(f"Error loading SVM model: {e}")


    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return
        preprocessed_image = self.preprocess(cv_image)


        direction = self.predict_direction(preprocessed_image)


        print("Arrow Direction:", direction)

    def preprocess(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (225, 225))  # Update size based on your training data
        img_blur = cv2.GaussianBlur(img_resized, (5, 5), 1)
        img_canny = cv2.Canny(img_blur, 50, 50)
        kernel = np.ones((3, 3))
        img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
        img_erode = cv2.erode(img_dilate, kernel, iterations=1)
        return img_erode


    def predict_direction(self, img):
        # Extract features from the preprocessed image (resize as needed)
        features = img.flatten()

        # Use the SVM model to predict the direction
        direction = self.svm_model.predict([features])[0]

        return direction

if __name__ == '__main__':
    svm_model_pathh = "/home/vignesh/Documents/svm_mars/svm_model.pkl"  
    ad_ros = ArrowDetectorROS(svm_model_pathh)
    rospy.spin()
