import numpy as np

def get_quantile_vector(vector, num_quantiles):
    quantiles = np.linspace(0, 1, num_quantiles+1)[1:-1]  # Exclude min and max quantiles
    quantile_vector = np.quantile(vector, quantiles)
    output_vector = np.searchsorted(quantile_vector, vector, side='right') / num_quantiles
    
    return output_vector