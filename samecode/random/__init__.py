import os
import random 
import tensorflow as tf
import numpy as np 
import torch

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED']=str(seed)
    
    try:
        tf.random.set_seed(seed)
    except:
        pass

    try:
        torch.manual_seed(seed)
    except:
        pass
    

    np.random.seed(seed)
    random.seed(seed)