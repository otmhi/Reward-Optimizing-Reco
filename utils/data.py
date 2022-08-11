from typing import List
import tensorflow as tf
import pandas as pd
from tqdm.auto import tqdm
import copy


class Dataset:
    def __init__(self, data:list):
        self.data = data
    
    def __getitem__(self, i):
        return self.data[i]
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return str(self.data[:3])
    
    def append(self, sample):
        self.data.append(sample)
        return
    
    def empty(self):
        self.data = []
        return
    
    def filter(self, f):
        return Dataset(list(filter(f, self.data)))
    
    def map(self, f):
        return Dataset(list(map(f, self.data)))


class Dataloader():
    def __init__(self, dataset:Dataset, batch_size:int, filters=[], transforms=[]):
        self.dataset = copy.deepcopy(dataset) # creates a copy of the dataset
        self.batch_size = batch_size
        self.filters = filters
        self.transforms = transforms
        
        for f in self.filters:
            self.dataset = self.dataset.filter(f)
    
        for t in self.transforms:
            self.dataset = self.dataset.map(t)
    
    def __len__(self): return int(len(self.dataset) / self.batch_size)
    
    def generator(self, **kwargs):
        for i in range(0, len(self.dataset), self.batch_size):
            lines = self.dataset[i:i+self.batch_size]
            batch = {'labels':0, 'context':{}, 'products':{} }
            
            # Slate size
            batch['slate_size'] = tf.concat([
                l['slate_size'] for l in lines
            ], axis=0)
            
            # Labels - padding of 0
            max_len = max([l['labels'].shape[-1] for l in lines])
            batch['labels'] = tf.concat([
                tf.concat([l['labels'], tf.zeros((1, max_len - l['labels'].shape[-1]), l['labels'].dtype)], -1)
                for l in lines
            ], axis=0)
            
            # Bidding features
            batch['bidding'] = tf.concat([
                l['bidding'] for l in lines
            ], axis=0)
            
            # Products padding -1
            max_len = max([l['products'].shape[-1] for l in lines])
            batch['products']['ppk'] = tf.concat([
                tf.concat([l['products'], tf.ones((1, max_len - l['products'].shape[-1]), l['products'].dtype)*(-1)], -1)
                for l in lines
            ], axis=0)
            
            batch['context']['gs'] = tf.ragged.constant([
                l['gs'].numpy().flatten() for l in lines
            ])
            
            batch['slate_propensity'] = tf.concat([
                l['slate_propensity'] for l in lines
            ], axis=0)
            
            batch['item_propensity'] = tf.concat([
                tf.concat([l['item_propensity'], tf.ones((1, max_len - l['item_propensity'].shape[-1]), l['item_propensity'].dtype)*(-1)], -1)
                for l in lines
                
            ], axis=0)
        
            yield batch
    
    
    def __call__(self, **kwargs):
        return self.generator(**kwargs)


def logs_dataframe(logs:List[dict]):
    columns = logs[0].keys()
    df = pd.DataFrame(columns=columns)

    def get_value(v):
        if tf.is_tensor(v):
            return float(v) if len(v.numpy().flatten().tolist()) == 1 else v.numpy().flatten()
        return v

    for i in tqdm(range(len(logs))):
        df = df.append({ k: get_value(v) for k,v in logs[i].items() }, ignore_index=True)
    return df