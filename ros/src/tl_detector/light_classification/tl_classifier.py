import os
import cv2
import math
import numpy as np
import tensorflow as tf
#from keras.models import load_model
from keras.models import model_from_json 
from styx_msgs.msg import TrafficLight

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #self.model = load_model(DIR_PATH + '/model.h5')
        json_file = open(DIR_PATH +'/model.json', 'r')
        model_json = json_file.read()
        self.model = model_from_json(model_json)
        self.model.load_weights(DIR_PATH + '/model.h5')
        self.model._make_predict_function()
        
        self.light_state = TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        resized_image = np.array([cv2.resize(hsv_image, (400, 300))])
        model_predict = self.model.predict(resized_image)
        predicted_state = int(model_predict.argmax(axis=-1))
        
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
