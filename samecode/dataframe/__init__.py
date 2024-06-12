import pandas as pd 

def transpose(df, index=None, label='sample_id'):
    
    columns = df[index]
    tdf = df.T
    
    tdf.columns = columns
    tdf.reset_index(inplace=True)
    tdf = tdf.rename(columns={'index': label})
    return tdf[tdf[label] != index]