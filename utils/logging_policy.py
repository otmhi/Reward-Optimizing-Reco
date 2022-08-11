import tensorflow as tf
import numpy as np

class LoggingPolicy:
    def __init__(self, catalog_size:int, **kwargs):
        self.catalog_size = catalog_size
        self.policy = np.array([
            [.33, .10, .07, .05, .12, .06, .27],
            [.24, .06, .12, .03, .09, .14, .32],
            [.15, .11, .03, .04, .27, .17, .23],
            [1/7]*7
        ])
    
    def recommend(self, ind_segments, slate_size:int, **kwargs):
        policy = self.policy[ind_segments.numpy().item()]
        return policy, 1+tf.constant(
            np.random.choice(self.catalog_size, p=policy, size=(1, slate_size), replace=False)
        )