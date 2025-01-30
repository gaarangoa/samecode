import os
import random 
import numpy as np 


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED']=str(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass

    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass
    
    np.random.seed(seed)
    random.seed(seed)