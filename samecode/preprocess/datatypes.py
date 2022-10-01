from collections import Counter 
import numpy as np 
import pandas as pd 

def whattype(key, dataset):
    
    '''This function identifies the data types for each column in a dataframe'''

    i = dataset[key]
    mx = pd.isna(i)
    i = i[~mx]
    
    if i.dtype.name == 'int64':
        if len(Counter(i)) == 2:
            return dict(
                variable=key, 
                dtype='binary', 
                categories=list(Counter(i).keys()),
                missing=np.sum(mx),
            )
        else:
            return dict(variable=key, dtype='numerical', categories=[], missing=np.sum(mx))
        
    elif i.dtype.name == 'float64':
        if len(Counter(i)) == 2:
            return dict(variable=key, dtype='binary', categories=list(Counter(i).keys() ), missing=np.sum(mx))
        else:
            return dict(variable=key, dtype='numerical', categories=[], missing=np.sum(mx))
    
    elif i.dtype.name == 'object':
        return dict(variable=key, dtype='categorical', categories=list(Counter(i).keys() ), missing=np.sum(mx))
    
    else:
        return 'unknown'


def is_unique(df):
    binaries = []
    for i in df.columns:
        val = df[i].dropna()
        items = list(Counter(val).keys())
        
        if len(items) == 1: 
            binaries.append(i)
            
    return binaries

def is_binary(df):
    binaries = []
    for i in df.columns:
        val = df[i].dropna()
        items = list(Counter(val).keys())
        
        if len(items) == 2: 
            binaries.append(i)
            
    return binaries


def nan_counts(df):
    f = []
    for col in df.columns:
        f.append({'variable': col, 'counter':np.sum(df[col].isnull()), 'percent': 100 * np.sum(df[col].isnull()) / df.shape[0]})
        
    return pd.DataFrame(f).sort_values('counter', ascending=False)

def nan_zero_counts(df):
    f = []
    for col in df.columns:
        _c = Counter(df[col].fillna(0))
        _r = [v/df.shape[0] for k,v in _c.items()]
        
        
        f.append({'variable': col, 'unique_values': len( _c ), 'val': _c, 'max_var_freq': np.max(_r)})
        
    return pd.DataFrame(f).sort_values('unique_values', ascending=True)