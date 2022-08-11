import tensorflow as tf
import matplotlib.pyplot as plt


def rank_label(l):
    l['labels'] = l['labels'][:,1:]
    return l


def xavier_init(shape:list):
    return tf.random.normal(shape, mean=0.0, stddev=float(len(shape)/sum(shape)),
                            dtype=tf.float32, seed=None, name=None)


def grad(model, criterion, batch):
    with tf.GradientTape() as tape:
        logits = model(ind_products=batch['products']['ppk'],
                       ind_segments=batch['context']['gs'],
                       bidding_features=batch['bidding'])
        loss_value = criterion(batch['labels'], logits)
    return loss_value, logits, tape.gradient(loss_value, model.trainable_variables)

def train_step(model, criterion, optimizer, batch):
    loss_value, output, grads = grad(model, criterion, batch)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value, output


def plot_loss(model):
    plt.plot(model.epoch_loss_values)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Loss {type(model).__name__}')


def plot_bidding(true_model:tf.Module, target_model:tf.Module, xmin=0.8, xmax=1.2):
    try:
        plt.scatter(
            true_model.bidding_weight.numpy().flatten(),
            target_model.bidding_weight.numpy().flatten()
        )
        plt.title("Bidding weights")
        plt.xlabel("true model")
        plt.ylabel(type(target_model).__name__)
    except:
        print("Target model doesn't have bidding_weight")
    
    plt.plot([xmin, xmax], [xmin, xmax], linestyle='--', linewidth=1, color='r')


def plot_product_scores(true_model:tf.Module, target_model:tf.Module, xmin=-2, xmax=2):
    plt.scatter(
        (true_model.product_lookup[1:,:] @ tf.transpose(true_model.context_lookup)).numpy().flatten(),
        (target_model.product_lookup[1:,:] @ tf.transpose(target_model.context_lookup)).numpy().flatten()
    )
    plt.title("Product/Context score")
    plt.xlabel("true model")
    plt.ylabel(type(target_model).__name__)
    plt.plot([xmin, xmax], [xmin, xmax], linestyle='--', linewidth=1, color="r")


def best_products(model: tf.Module, ind_segments:tf.Tensor, verbose=True):
    """
    ind_segments: tf.Tensor [N, GC] / GC: total number of grapeshot categories
    """
    try:
        context_embeddings = model.get_context_embeddings(ind_segments)
        score = tf.math.softmax(
            tf.squeeze(
                tf.expand_dims(context_embeddings, axis=1) @ tf.transpose(model.product_lookup[1:,:]),
                axis=1)
        )
        sorted_index = tf.argsort(score, direction="DESCENDING").numpy().reshape(-1)
        if verbose:
            print(f"GS categories: {ind_segments}")
            print(f"Best products: {sorted_index+1}")
            print(f"Product scores: {tf.math.exp(score).numpy().reshape(-1)[sorted_index].tolist()}")
        return sorted_index+1, score.numpy().flatten().round(3)
    except:
        return [None]