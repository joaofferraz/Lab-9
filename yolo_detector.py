from tensorflow.keras.models import load_model
import cv2
import numpy as np
from utils import sigmoid

class YoloDetector:
    """
    Represents an object detector for robot soccer based on the YOLO algorithm.
    """
    def __init__(self, model_name, anchor_box_ball=(5, 5), anchor_box_post=(2, 5)):
        """
        Constructs an object detector for robot soccer based on the YOLO algorithm.

        :param model_name: name of the neural network model which will be loaded.
        :type model_name: str.
        :param anchor_box_ball: dimensions of the anchor box used for the ball.
        :type anchor_box_ball: bidimensional tuple.
        :param anchor_box_post: dimensions of the anchor box used for the goal post.
        :type anchor_box_post: bidimensional tuple.
        """
        self.network = load_model(model_name + '.hdf5')
        self.network.summary()  # prints the neural network summary
        self.anchor_box_ball = anchor_box_ball
        self.anchor_box_post = anchor_box_post

    def detect(self, image):
        """
        Detects robot soccer's objects given the robot's camera image.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        # Todo: implement object detection logic

        output = self.network.predict(self.preprocess_image(image))
        ball_detection, post1_detection, post2_detection = self.process_yolo_output(output)
        
        return ball_detection, post1_detection, post2_detection

    def preprocess_image(self, image):
        """
        Preprocesses the camera image to adapt it to the neural network.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: image suitable for use in the neural network.
        :rtype: NumPy 4-dimensional array with dimensions (1, 120, 160, 3).
        """
        # Todo: implement image preprocessing logic
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = image / 255
        image = np.reshape(image, (1, 120, 160, 3))

        return image

    def process_yolo_output(self, output):
        """
        Processes the neural network's output to yield the detections.

        :param output: neural network's output.
        :type output: NumPy 4-dimensional array with dimensions (1, 15, 20, 10).
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        coord_scale = 4 * 8  # coordinate scale used for computing the x and y coordinates of the BB's center
        bb_scale = 640  # bounding box scale used for computing width and height
        output = np.reshape(output, (15, 20, 10))  # reshaping to remove the first dimension
        # Todo: implement YOLO logic

        max_prob = [0, 0, 0]
        line = [0, 0, 0]
        column =[0, 0, 0]

        for i in range(15):
            for j in range(20):
                if (sigmoid(output[i][j][0]) > max_prob[0]):
                    max_prob[0] = sigmoid(output[i][j][0])
                    line[0] = i
                    column[0] = j
                if (sigmoid(output[i][j][5]) > max_prob[1]):
                    max_prob[1] = sigmoid(output[i][j][5])
                    line[1] = i
                    column[1] = j
                elif (sigmoid(output[i][j][5]) > max_prob[2]):
                    max_prob[2] = sigmoid(output[i][j][5])
                    line[2] = i
                    column[2] = j

        x_ball = (column[0] + sigmoid(output[line[0]][column[0]][1])) * coord_scale
        y_ball = (line[0] + sigmoid(output[line[0]][column[0]][2])) * coord_scale
        w_ball = bb_scale * self.anchor_box_ball[0] * np.exp(output[line[0]][column[0]][3]) 
        h_ball = bb_scale * self.anchor_box_ball[1] * np.exp(output[line[0]][column[0]][4]) 

        x_post1 = (column[1] + sigmoid(output[line[1]][column[1]][6])) * coord_scale
        y_post1 = (line[1] + sigmoid(output[line[1]][column[1]][7])) * coord_scale
        w_post1 = bb_scale * self.anchor_box_post[0] * np.exp(output[line[1]][column[1]][8]) 
        h_post1 = bb_scale * self.anchor_box_post[1] * np.exp(output[line[1]][column[1]][9])

        x_post2 = (column[2] + sigmoid(output[line[2]][column[2]][6])) * coord_scale
        y_post2 = (line[2] + sigmoid(output[line[2]][column[2]][7])) * coord_scale
        w_post2 = bb_scale * self.anchor_box_post[0] * np.exp(output[line[2]][column[2]][8]) 
        h_post2 = bb_scale * self.anchor_box_post[1] * np.exp(output[line[2]][column[2]][9])

        ball_detection = (max_prob[0], x_ball, y_ball, w_ball, h_ball)  # Todo: change this line
        post1_detection = (max_prob[1], x_post1, y_post1, w_post1, h_post1)  # Todo: change this line
        post2_detection = (max_prob[2], x_post2, y_post2, w_post2, h_post2)  # Todo: change this line
        
        return ball_detection, post1_detection, post2_detection
