import numpy as np
import pandas as pd
from .simulator import Simulator
import tensorflow as tf
from tqdm.auto import tqdm

class ABTest:
    def __init__(self, simulator:Simulator, **models):
        self.simulator = simulator
        self.models = models
        self.logs = []
    
    def logits_to_label(self, logits):
        label = tf.random.categorical(logits, 1, dtype=tf.int32)
        return tf.squeeze(tf.one_hot(label, depth=logits.shape[1]), 0)

    def label_reward(self, logits):
        return float(1 - tf.nn.softmax(logits)[0][0])
    
    def __call__(self, nb_samples=10_000, with_faiss=False, **kwargs):
        logs = []
        for _ in tqdm(range(nb_samples)):
            log = {}
            banner = self.simulator.sample_session()
            log['slate_size'] = banner['slate_size'].numpy().item()
            log['gs'] = banner['gs'].numpy().flatten()
            log['bidding'] = banner['bidding'].numpy().flatten()
            
            for name, model in self.models.items():
                slate_size = banner['slate_size'].numpy().item()
                if with_faiss:
                    _, action = model.recommend_faiss(ind_segments=banner['gs'], slate_size=slate_size)
                else: 
                    _, action = model.recommend(ind_segments=banner['gs'], slate_size=slate_size)
                
                logits = self.simulator.oracle_model(ind_products=action, ind_segments=banner['gs'],
                                                     bidding_features=banner['bidding'], **banner)
                log[f'action_{name}'] = action.numpy().flatten()
                log[f'reward_{name}'] = self.label_reward(logits)
                log[f'label_{name}'] = self.logits_to_label(logits)
            logs.append(log)
            
        self.logs = pd.DataFrame(logs)
        ctr = np.mean([self.simulator.logs[i]['reward'] for i in range(len(self.simulator.logs))])
        std = round(np.sqrt(ctr*(1-ctr)/nb_samples)*100, 2)
        ctr = round(ctr*100,2)
        print(f"CTR Logs: {ctr} - {std} %")

        for name, _ in self.models.items():
            ctr = self.logs[f'reward_{name}'].mean()
            std = round(np.sqrt(ctr*(1-ctr)/nb_samples)*100, 2)
            ctr = round(ctr*100,2)
            print(f"CTR {name}: {ctr} - {std} %")
        
        return self.logs