#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from train import Trainer
import numpy as np
# Initialize trainer
# trainer = Trainer(force_cpu=False)


class Artifitial_Strategy(object):

    def __init__(self):
        rospy.init_node("artifitial_strategy_node", anonymous=True)
        print("Init.")
        rospy.Subscriber("/predict_img", Float64MultiArray, self.predict_img_callback)

        self.iteration = 0
        self.prev_grasp_success = None
        self.best_pix_ind = None
        self.best_pose = []
        self.predict_img = np.zeros([255,255])
    def callback(self, msg):
        # x = msg.data TODO: Data change
        grasp_predictions, state_feat = trainer.forward(predict_heightmap, is_volatile=True)
        # use_heuristic = True
        self.best_pix_ind = trainer.grasp_heuristic(predict_heightmap)
        predicted_value = grasp_predictions[self.best_pix_ind]
        # use_heuristic = False
        self.best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        predicted_value = np.max(grasp_predictions)

        self.best_pose[0] = self.best_pix_ind[2]
        self.best_pose[1] = self.best_pose[1]
        self.best_pose[2] = np.deg2rad(self.best_pix_ind[0]*(360/trainer.model.num_rotations))
        # back
        # Compute training labels
        label_value, prev_reward_value = trainer.get_label_value(grasp_sucess, change_detected, prev_grasp_predictions, next_predict_heightmap)

    def predict_img_callback(self, msg):
        for i in range(255):
            for j in range(255):
                self.predict_img[i,j] = msg.data[i*255+j]
        print("...")


if __name__ == '__main__':
    artifitial_strategy_node = Artifitial_Strategy()
    print("ros.spin()")
    rospy.spin()