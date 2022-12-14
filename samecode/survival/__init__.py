from lifelines import CoxPHFitter
from collections import Counter
import pandas as pd

def extract_kmstats(data, features, time, event):
    '''
    Extract a table with km statistics on a given set of clusters or features present in the data
    '''
    resc = []
    data_ = data.copy()
    
    for cluster in features:
        # Run a cox ph to get the Hazard ratio and statistics
        try:
            cph = CoxPHFitter().fit(data_[[time, event, cluster]], time, event)
            
            # Summary
            sm = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].reset_index(drop=False)
            sm['nP'] = Counter(data_[cluster])[1]
            sm['nW'] = Counter(data_[cluster])[0]
            sm.columns = ['variable', 'hr', 'hr_lo', 'hr_hi', 'pval', 'nN', 'nW']
            
            resc.append(sm)
            
        except Exception as inst:
            print(inst)
            continue

        
        
    return pd.concat(resc)
        