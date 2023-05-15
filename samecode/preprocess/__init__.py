def split_vector(N, k):
    # Calculate the number of partitions required
    partitions = N // k
    remaining = N % k
    
    result = []
    start = 0
    
    # Split the vector into smaller vectors
    for i in range(partitions):
        # Calculate the end index of the current partition
        end = start + k
        
        # Add the (start, end) index pair to the result
        result.append((start, end))
        
        # Update the start index for the next partition
        start = end
    
    # If there are remaining elements, add an extra partition
    if remaining > 0:
        end = start + remaining
        result.append((start, end))
    
    return result