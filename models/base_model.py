import tensorflow as tf
import os
import sys
import numpy as np
class BaseModel:
    def __init__(self, sess):
        self.sess = sess
        self.weights = []
        self.weight_phs = []
        self.set_op = []
        self.value_set_op = []

    def get_weights(self):
        return self.sess.run(self.weights)

    def set_weights(self, new_weights):
        self.sess.run(self.set_op, feed_dict=dict(zip(self.weight_phs, new_weights)))

