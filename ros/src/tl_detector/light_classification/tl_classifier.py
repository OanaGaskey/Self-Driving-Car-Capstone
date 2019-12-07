import os
import cv2
import math
import numpy as np
# import tensorflow as tf
# from keras.models import model_from_json 
from styx_msgs.msg import TrafficLight



DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class TLClassifier(object):
    def __init__(self):
        # load classifier
        #self.model = load_model(DIR_PATH + '/model.h5')
        # json_file = open(DIR_PATH +'/model.json', 'r')
        # model_json = json_file.read()
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True
        # session = tf.compat.v1.InteractiveSession(config=config)

        # tf.compat.v1.keras.backend.set_session(session)

        # self.model = model_from_json(model_json)
        # self.model.load_weights(DIR_PATH + '/model.h5')
        # self.model._make_predict_function()

        # self.graph = tf.get_default_graph()
        self.light_buffer = np.array([0, 0, 0])
        self.dist = 10
        self.light_state = TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        
        # resized_image = np.array([self.filter_redlight(image)])
        # if np.sum(resized_image)>10000:
        #     predicted_state=TrafficLight.RED
        # else:
        #     predicted_state=TrafficLight.UNKNOWN
        # resized_image = np.array([hsv_image])

        # if self.model:
        #     if len(resized_image)>0:

        # with self.graph.as_default():
        #     model_predict = self.model.predict(resized_image)
        #     predicted_state = int(model_predict.argmax(axis=-1))
        #     print(str(predicted_state))

        mask_sum, label = self.classify_tl(image)
        if self.dist > 0 :
            self.light_buffer = np.add(self.light_buffer, mask_sum)
            predicted_state = np.argmax(self.light_buffer)
            self.dist += -1
        else:
            predicted_state = label
            self.light_buffer = mask_sum
            
            self.dist=10

        # predicted_state = label

        print(self.light_buffer)
        print(self.dist)
        print(str(mask_sum))

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

    # def filter_redlight(self, image):
    #     resized_image = cv2.resize(image, (400, 300))
    #     img_hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    #     # lower mask (0-10)
    #     lower_red = np.array([0,50,50])
    #     upper_red = np.array([10,255,255])
    #     mask = cv2.inRange(img_hsv, lower_red, upper_red)

    #     # or your HSV image, which I *believe* is what you want
    #     # output_hsv = img_hsv.copy()
    #     # output_hsv[np.where(mask==0)] = 0
    #     return mask

    def classify_tl(self, image):
        img_hsv = cv2.cvtColor(image[0:300, 0:500], cv2.COLOR_BGR2HSV)
            
        # lower mask (0-10)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])

        lower_green = np.array([50,50,50])
        upper_green = np.array([70,255,255])

        lower_yellow = np.array([20,150,150])
        upper_yellow = np.array([30,255,255])
        
        mask = np.array([cv2.inRange(img_hsv, lower_red, upper_red),
                        cv2.inRange(img_hsv, lower_yellow, upper_yellow),
                        cv2.inRange(img_hsv, lower_green, upper_green)])
        
        
        mask_sum = np.sum(np.sum(mask, axis=1), axis=1)
        label = np.argmax(mask_sum)
        
        # if mask_sum[label] < 5:
        #     label = 4
            

        # or your HSV image, which I *believe* is what you want
    #     output_hsv = img_hsv.copy()
    #     output_hsv[np.where(mask==0)] = 0

        return mask_sum, label
