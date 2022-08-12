import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from utils.data import Dataloader
from utils.train import xavier_init

class IndependentIPS(tf.Module):
    """Just a reco model trained with the IPS method"""
    
    def __init__(self, nb_gs_categories, embedding_dim, catalog_size, **kwargs):
        self.context_lookup = tf.Variable(xavier_init([nb_gs_categories, embedding_dim]),
                                          trainable=True)
        self.product_lookup = tf.Variable(xavier_init([catalog_size+1, embedding_dim]),
                                          trainable=True)
        self.epoch_loss_values = []

    def get_context_embeddings(self, ind_segments):
        return tf.reduce_mean(tf.nn.embedding_lookup(self.context_lookup, ind_segments), axis=1)

    def recommend(self, ind_segments, slate_size:int, **kwargs):
        """Returns indices of the recommended catalog items"""
        context_embeddings = self.get_context_embeddings(ind_segments)

        policy = tf.squeeze(
            tf.expand_dims(context_embeddings, axis=1) @ tf.transpose(self.product_lookup[1:,:]), # remove the padding
            axis=1
        )
        action = tf.argsort(policy, direction="DESCENDING")[:,:slate_size] + 1
        return policy, action
    
    def train(self, dataset, num_epochs:int, batch_size:int, optimizer, topk=0, **kwargs):
        dataloader = Dataloader(dataset, batch_size, filters=[
            lambda l: l["labels"][0,0] == 0 # keep only clicked banners
        ])
        
        for _ in tqdm(range(num_epochs)):
            epoch_loss = []
            for _, batch in enumerate(dataloader()):
                loss_value = item_ips_train_step(self, optimizer, batch, topk)
                epoch_loss.append(loss_value)

            self.epoch_loss_values.append(np.sum(epoch_loss))


def get_model_item_propensity(model, ind_segments, ind_item):
    context_embeddings = model.get_context_embeddings(ind_segments)
    product_scores = tf.squeeze(
        tf.expand_dims(context_embeddings, axis=1) @ tf.transpose(model.product_lookup[1:,:])
        , axis=1
    )
    policy = tf.nn.softmax(product_scores)
    batch_size = ind_segments.shape[0]
    policy = tf.concat([ tf.ones((batch_size,1)), policy ], axis=-1) # add padding
    
    return tf.concat([
        tf.expand_dims(tf.gather(policy[i], ind_item[i]), axis=0) for i in range(batch_size)
    ], axis=0)


def item_ips_train_step(model, optimizer, batch, topk:int):
    """ Example:
    slate: [2, 10, 3]
    label: [0, 1, 0, 0]
    ips loss: - model_policy[2] / logging_policy[2]
    """
    # Clicked items
    click_position = tf.argmax(batch['labels'], output_type=tf.int32, axis=1) - 1
    batch_size = batch['labels'].shape[0]
    click_position_nd = tf.transpose(tf.concat([
        tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=0),
        tf.expand_dims(click_position, axis=0)
    ], axis=0))
    ind_item = tf.gather_nd(batch['products']['ppk'], click_position_nd)
    
    # Logging propensity
    logging_propensity = tf.gather_nd(batch['item_propensity'], click_position_nd)
      
    # TopK Heuristic
    lamb = 1
    if topk:
        lamb = topk * (
            1 - get_model_item_propensity(model, batch['context']['gs'], ind_item)
        ) ** (topk-1)

    # IPS Estimator
    with tf.GradientTape() as tape:
        # Model propensity
        model_propensity = get_model_item_propensity(model, batch['context']['gs'], ind_item)

        loss = - tf.reduce_mean(
            lamb * model_propensity / logging_propensity
        )
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss