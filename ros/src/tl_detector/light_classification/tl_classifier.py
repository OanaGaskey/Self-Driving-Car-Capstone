import os
import cv2
import math
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        self.light_buffer = np.array([0, 0, 0])
        self.dist = 5
        self.light_state = TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        mask_sum, label = self.classify_tl(image)
        if self.dist > 0 :
            self.light_buffer = np.add(self.light_buffer, mask_sum)
            predicted_state = np.argmax(self.light_buffer)
            self.dist += -1
        else:
            predicted_state = label
            self.light_buffer = mask_sum
            
            self.dist=5
            
        print(self.light_buffer)

        if predicted_state == 0:
            print("RED")
            self.light_state = TrafficLight.RED
        elif predicted_state == 1:
            print("YELLOW")
            self.light_state = TrafficLight.YELLOW
        elif predicted_state == 2:
            print("GREEN")
            self.light_state = TrafficLight.GREEN
        else:
            print("UNKNOWN")
            self.light_state = TrafficLight.UNKNOWN
        
        return self.light_state

    def classify_tl(self, image):
        img_hsv = cv2.cvtColor(image[0:600, 0:500], cv2.COLOR_BGR2HSV)

        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        
        lower_yellow = np.array([20,150,150])
        upper_yellow = np.array([30,255,255])

        lower_green = np.array([50,100,100])
        upper_green = np.array([70,255,255])
        
        mask = np.array([cv2.inRange(img_hsv, lower_red, upper_red),
                        cv2.inRange(img_hsv, lower_yellow, upper_yellow),
                        cv2.inRange(img_hsv, lower_green, upper_green)])
        
        
        mask_sum = np.sum(np.sum(mask, axis=1), axis=1)
        label = np.argmax(mask_sum)
        
        return mask_sum, label
