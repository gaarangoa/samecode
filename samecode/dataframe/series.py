import numpy as np

def get_quantile_vector(vector, num_quantiles):
    """
    Divides a vector into a specified number of quantiles and maps its elements 
    to normalized quantile positions.

    Args:
        vector (numpy.ndarray): The input vector of numerical data.
        num_quantiles (int): The desired number of quantiles.

    Returns:
        numpy.ndarray: An array representing quantile positions for each element 
                       in the original vector, with values normalized between 0 and 1. 
                       A value of 0 indicates the lowest quantile, and 1 the highest quantile.
    """

    quantiles = np.linspace(0, 1, num_quantiles+1)[1:-1]  # Exclude min and max quantiles
    quantile_vector = np.quantile(vector, quantiles)
    output_vector = np.searchsorted(quantile_vector, vector, side='right') / num_quantiles
    
    return output_vector


def get_quantile_distribution(vector, num_quantiles):

    quantiles = np.linspace(0, 1, num_quantiles)  # Exclude min and max quantiles
    quantile_vector = np.quantile(vector, quantiles)
    
    return quantile_vector