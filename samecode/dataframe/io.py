from operator import itemgetter
from tqdm.auto import tqdm
import pandas as pd

def load_selected_columns(fi, selected_fields=[]):
    '''
    Given a very large file load a particular set of columns. 
    fi: Input file
    selected_fields: Column names to extract from the file 

    returns a pandas dataframe with the selected values
    '''
    clp = []
    for ix, i in tqdm(enumerate(open(fi))):
        if ix == 0:
            fields = {i:ix for ix,i in enumerate(i.strip().split(','))}
            item_ix = [fields[i] for i in selected_fields]
        else:
            item = i.strip().split(',')
            clp.append({selected_fields[ix]: item[item_ix[ix]] for ix,i in enumerate(itemgetter(*item_ix)(item))})

    return clp


def load_selected_columns_unique_entries(fi, keys=[], values=[], sep=','):
    '''
    Given a very large file load a particular set of columns. 
    fi: Input file
    selected_fields: Column names to extract from the file 

    returns a pandas dataframe with the selected values
    '''
    clp = {}
    for ix, i in tqdm(enumerate(open(fi))):
        if ix == 0:
            fields = {i:ix for ix,i in enumerate(i.strip().split(sep))}
            keysx = [fields[i] for i in keys]
            valuesx = [fields[i] for i in values]
        else:
            item = i.strip().split(',')
            clp[tuple([item[k] for k in keysx])] = [item[k] for k in valuesx]
        
    return pd.DataFrame([list(k)+v for k,v in clp.items()], columns=keys+values)