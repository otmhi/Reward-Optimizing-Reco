import numpy as np
import tensorflow as tf

class RandomModel(tf.Module):
    def __init__(self, catalog_size, **kwargs):
        self.catalog_size = catalog_size
    
    def recommend(self, ind_segments, slate_size:int, **kwargs):
        policy = np.ones(self.catalog_size)/self.catalog_size
        return policy, 1 + tf.constant(
            np.random.choice(self.catalog_size, p=policy, size=(1, slate_size), replace=False),
            dtype=tf.int32)