import scipy.stats
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from .data import Dataset

class Simulator:
    def __init__(self, oracle_model, logging_model, max_slate_size:int, min_slate_size:int,
                 catalog_size:int, nb_bidding_features:int, nb_gs_categories:int,
                 max_gs_categories=10, **kwargs):
        
        self.oracle_model = oracle_model
        self.logging_model = logging_model        
        self.max_slate_size = max_slate_size
        self.min_slate_size = min_slate_size
        self.nb_bidding_features = nb_bidding_features
        self.nb_gs_categories = nb_gs_categories
        self.catalog_size = catalog_size
        self.max_gs_categories = max_gs_categories        
        self.logs = Dataset([])
    
    
    def sample_session(self):
        if self.max_slate_size == 1:
            slate_size = np.array([1])
        else:
            slate_size = np.array([np.random.randint(self.min_slate_size, self.max_slate_size+1)])
        bidding_features = np.random.rand(self.nb_bidding_features)
        # sample grapheshot
        nb_categories = min(1 + scipy.stats.distributions.poisson.rvs(3), self.max_gs_categories)
        gs = np.random.choice(self.nb_gs_categories, replace=False, size=nb_categories)
        return dict(
            slate_size=tf.constant([slate_size], tf.int32),
            bidding=tf.constant([bidding_features], tf.float32),
            gs=tf.constant([gs], tf.int32)
        )
    
    
    def sample_action(self, slate_size:int, gs, **kwargs):
        """Returns the indices of the recommended items (pi distribution)"""
        logging_policy, action = self.logging_model.recommend(ind_segments=gs, slate_size=slate_size.numpy().item())
        full_slate_propensity, single_item_propensity = self.get_logging_propensity(logging_policy, action)
        return action, full_slate_propensity, single_item_propensity
    

    def get_logging_propensity(self, logging_policy, action):
        p = []
        for item in action[0]:
            p.append(logging_policy[item-1] / (1 - sum(p)))
        full_slate_propensity = tf.constant([np.prod(p)], tf.float32)
        single_item_propensity = tf.constant([logging_policy[action-1]], tf.float32)
        return full_slate_propensity, single_item_propensity
    

    def simulate_logs(self, nb_samples, **kwargs) -> Dataset:
        """Simulate logs"""
        self.logs.empty()
        
        for _ in tqdm(range(nb_samples)):
            sample = self.sample_session()
            sample['products'], sample['slate_propensity'], sample['item_propensity'] = self.sample_action(**sample)
            logits = self.oracle_model(ind_products=sample['products'],
                                       ind_segments=sample['gs'], bidding_features=sample['bidding'])
            label = tf.random.categorical(logits, 1, dtype=tf.int32)
            sample['labels'] = tf.squeeze(tf.one_hot(label, depth=logits.shape[1]), 0)
            sample['reward'] = float(1 - tf.nn.softmax(logits)[0][0])
            self.logs.append(sample)
        return self.logs

    def __call__(self, nb_samples, **kwargs):
        return self.simulate_logs(nb_samples=nb_samples, **kwargs)
            

def logits_to_label(logits):
    label = tf.random.categorical(logits, 1, dtype=tf.int32)
    return tf.squeeze(tf.one_hot(label, depth=logits.shape[1]), 0)

