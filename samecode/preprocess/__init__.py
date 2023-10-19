import numpy as np 

def binx(x, bins):
    hist, bin_edges = np.histogram(x, bins=bins)
    return [bin_edges[i-1] for i in np.digitize(x, bin_edges[:-1])]

def split_vector(N, k):
    # Calculate the number of partitions required
    partitions = N // k
    remaining = N % k
    
    result = []
    start = 0
    index = 0
    # Split the vector into smaller vectors
    for i in range(partitions):
        # Calculate the end index of the current partition
        end = start + k
        
        # Add the (start, end) index pair to the result
        result.append((start, end, index))
        
        # Update the start index for the next partition
        start = end
        index += 1 
    
    # If there are remaining elements, add an extra partition
    if remaining > 0:
        end = start + remaining
        result.append((start, end, index))
    
    return result