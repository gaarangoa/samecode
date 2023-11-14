from lifelines import CoxPHFitter
from collections import Counter
import pandas as pd

def extract_kmstats(data, features, time, event, labels=[0, 1]):
    r"""Extract a table with km statistics on a given set of clusters or features present in the data

    Args:
        data (dataframe): pandas dataframe with the data.
        features (list): list of features to compute the KM stats.
        time (str): time column.
        event (str): event column.
    """

    resc = []
    data_ = data.copy()
    
    for cluster in features:
        # Run a cox ph to get the Hazard ratio and statistics
        try:
            cph = CoxPHFitter().fit(data_[[time, event, cluster]], time, event)
            
            # Summary
            sm = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].reset_index(drop=False)
            sm['n{}'.format(labels[1])] = Counter(data_[cluster])[1]
            sm['n{}'.format(labels[0])] = Counter(data_[cluster])[0]
            sm.columns = ['variable', 'hr', 'hr_lo', 'hr_hi', 'pval', 'n{}'.format(labels[1]), 'n{}'.format(labels[0])]
            
            resc.append(sm)
            
        except Exception as inst:
            print(inst)
            continue

        
        
    return pd.concat(resc)
        