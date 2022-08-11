import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from utils.data import Dataloader
from utils.train import xavier_init, train_step, rank_label

class RankingModel(tf.Module):
    def __init__(self, max_slate_size: int, nb_gs_categories: int, embedding_dim: int, catalog_size: int,
                 add_position_bias_init=None, mult_position_bias_init=None, **kwargs):
        self.context_lookup = tf.Variable(xavier_init([nb_gs_categories, embedding_dim]), trainable=True)
        self.product_lookup = tf.Variable(xavier_init([catalog_size+1, embedding_dim]), trainable=True)
        
        if not type(add_position_bias_init) == type(None):
            self.add_position_bias = tf.Variable(add_position_bias_init, trainable=False)
        else:
            self.add_position_bias = tf.Variable(xavier_init([1, max_slate_size]), trainable=True)    
        
        if not type(mult_position_bias_init) == type(None):
            self.mult_position_bias = tf.Variable(mult_position_bias_init, trainable=False)
        else:
            self.mult_position_bias = tf.Variable(xavier_init([1, max_slate_size]), trainable=True)    
        
        self.epoch_loss_values = []
    

    def get_context_embeddings(self, ind_segments):
        return tf.reduce_mean(tf.nn.embedding_lookup(self.context_lookup, ind_segments), axis=1)
    
    def get_product_embeddings(self, ind_products):
        return tf.nn.embedding_lookup(self.product_lookup, tf.maximum(ind_products, 0))
    
    def get_reco_scores(self, context_embeddings, product_embeddings, max_slate_size):
        product_score = tf.squeeze(
            tf.expand_dims(context_embeddings, axis=1) @ tf.transpose(product_embeddings, perm=[0,2,1])
            , axis=1) + self.mult_position_bias[:,:max_slate_size]
        return tf.math.log(
            tf.math.exp(product_score) + tf.math.exp(self.add_position_bias[:,:max_slate_size])
        )
    
    def __call__(self, ind_products, ind_segments, **kwargs):
        slate_size = tf.reduce_sum(tf.where(ind_products == -1, 0, 1), axis=1, keepdims=True)
        max_slate_size = tf.reduce_max(slate_size).numpy()
        # Get grapeshot embeddings
        context_embeddings = self.get_context_embeddings(ind_segments)
        # Get product embeddings
        product_embeddings = self.get_product_embeddings(ind_products)
        # Compute rank scores
        reco_scores = self.get_reco_scores(context_embeddings, product_embeddings, max_slate_size)
        return tf.where(ind_products == -1, -1e8, reco_scores)

    
    def recommend(self, ind_segments, slate_size:int, **kwargs):
        """Returns indices of the recommended catalog items"""
        context_embeddings = self.get_context_embeddings(ind_segments)
        policy = tf.squeeze(
            tf.expand_dims(context_embeddings, axis=1) @ tf.transpose(self.product_lookup[1:,:]), # remove the padding
            axis=1
        )
        action = tf.argsort(policy, direction="DESCENDING")[:,:slate_size] + 1
        return tf.nn.softmax(policy), action
    
    
    def train(self, dataset, num_epochs:int, batch_size:int, nb_samples:int, criterion, optimizer, **kwargs):        
        dataloader = Dataloader(dataset, batch_size,
                                filters=[
                                    lambda l: l["labels"][0,0] == 0 # keep only clicked banners
                                ],
                                transforms=[rank_label])
        
        for _ in tqdm(range(num_epochs)):
            epoch_loss = []
            for _, batch in enumerate(dataloader()):
                loss_value, _ = train_step(self, criterion, optimizer, batch)
                epoch_loss.append(loss_value)

            self.epoch_loss_values.append(np.sum(epoch_loss))