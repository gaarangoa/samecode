import os
import random 
import tensorflow as tf
import numpy as np 

def reset_random_seeds(seed=0):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)