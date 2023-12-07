#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ArrowDetectorROS:
    def __init__(self):
        rospy.init_node('arrow_detector', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return

        contours, hierarchy = cv2.findContours(self.preprocess(cv_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            hull = cv2.convexHull(approx, returnPoints=False)
            sides = len(hull)

            if 6 > sides > 3 and sides + 2 == len(approx):
                arrow_tip = self.find_tip(approx[:,0,:], hull.squeeze())
                if arrow_tip:
                    cv2.drawContours(cv_image, [cnt], -1, (0, 255, 0), 3)
                    cv2.circle(cv_image, arrow_tip, 3, (0, 0, 255), cv2.FILLED)

        cv2.imshow("Arrow Detection", cv_image)
        cv2.waitKey(1)

    def preprocess(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        img_canny = cv2.Canny(img_blur, 50, 50)
        kernel = np.ones((3, 3))
        img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
        img_erode = cv2.erode(img_dilate, kernel, iterations=1)
        return img_erode

    def find_tip(self, points, convex_hull):
        length = len(points)
        indices = np.setdiff1d(range(length), convex_hull)

        for i in range(2):
            j = indices[i] + 2
            if j > length - 1:
                j = length - j
            if np.all(points[j] == points[indices[i - 1] - 2]):
                return tuple(points[j])

if __name__ == '__main__':
    try:
        ad_ros = ArrowDetectorROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
