import numpy as np
import random 

SEED = 1024
np.random.seed(SEED)
random.seed(SEED)

def useYesterdayValue(data, probs=True):
    if probs:
        padding = [[0.0] * len(data[0])] 
    else:
        padding = [[0] * len(data[0])]
    new_data = padding + data[:-1]
    return new_data


def useAverageValue(data, days_back=5):
    n_days = len(data)
    results = []

    for i in range(n_days):
        if i == 0:
            if isinstance(data[0], (list, tuple)) and len(data[0]) > 0:
                if isinstance(data[0][0], (list, tuple)):
                    # Matches list of tuples: [[(0, 0)]]
                    results.append([tuple([0] * len(item)) for item in data[0]])
                else:
                    # Matches list of numbers: [0, 0]
                    results.append([0.0] * len(data[0]))
            else:
                results.append(0.0)
            continue

        start_idx = max(0, i - days_back)
        window = data[start_idx : i] 
        sample_day = window[0]
        
        # CASE A: TopER Structure - List of Tuples [[(nodes, edges)]]
        if isinstance(sample_day, (list, tuple)) and len(sample_day) > 0 and isinstance(sample_day[0], (list, tuple)):
            day_avg = []
            for tuple_group in zip(*window):
                # We use int(round(...)) because node/edge counts MUST be integers
                avg_tuple = tuple(int(round(sum(vals) / len(vals))) for vals in zip(*tuple_group))
                day_avg.append(avg_tuple)
            results.append(day_avg)

        # CASE B: Probabilities - List of Numbers [p1, p2, p3]
        elif isinstance(sample_day, (list, tuple)):
            # Probabilities should remain floats, so we don't round here
            day_avg = [sum(idx_group) / len(idx_group) for idx_group in zip(*window)]
            results.append(day_avg)
            
        # CASE C: Flat scalars
        else:
            results.append(sum(window) / len(window))

    return results

def randomTopER(num_vectors, max_nodes=1000, max_edges_limit=None, num_buckets=10):
    """
    Generates a monotonic TopER vector using random growth increments.
    Each bucket (n, e) is >= the previous bucket.
    """
    res = []
    
    for i in range(num_vectors):
        toper_vector = []
        
        # Initialize starting point (Bucket 1)
        curr_n = np.random.randint(1, 10) # Start with a small graph
        
        # Max edges for the starting nodes
        start_max_e = int(curr_n * (curr_n - 1) / 2) if curr_n > 1 else 0
        curr_e = np.random.randint(0, start_max_e + 1)
        
        toper_vector.append((curr_n, curr_e))
        
        # Generate the next 9 buckets
        for _ in range(num_buckets - 1):
            # 1. Randomly decide growth (can be 0 for a 'flat' filtration level)
            # We use a fraction of max_nodes to keep jumps reasonable
            n_jump = np.random.randint(0, max_nodes // 4) 
            curr_n += n_jump
            
            # 2. Calculate edge growth
            # We ensure e doesn't exceed the new theoretical max for the node count
            theoretical_max_e = int(curr_n * (curr_n - 1) / 2) if curr_n > 1 else 0
            
            if max_edges_limit:
                theoretical_max_e = min(theoretical_max_e, max_edges_limit)
                
            # Edge jump must be at least 0, but can't exceed the theoretical max
            max_possible_jump = max(0, theoretical_max_e - curr_e)
            e_jump = np.random.randint(0, max_possible_jump + 1) if max_possible_jump > 0 else 0
            curr_e += e_jump
            
            toper_vector.append((curr_n, curr_e))
            
        res.append(toper_vector)
    
    return np.array(res)

def randomProbs(num_vectors):
    res = []
    for i in range(num_vectors):
        # 1. Generate first part (2 indices)
        p = np.random.random()
        part_1 = [p, 1.0 - p]
        
        # 2. Generate second part (4 indices)
        raw_part_2 = np.random.rand(4)
        part_2 = (raw_part_2 / raw_part_2.sum()).tolist()
        
        # Combine and round for cleaner printing/Excel display
        combined = part_1 + part_2
        res.append(combined)

    # Convert the whole list to a 2D array at the end
    return np.array(res)

def ablationSetup(toper, probs, setting=0):
    if setting == 0:
        return toper, probs
    elif setting == 1:
        new_toper = useYesterdayValue(toper, False)
        return new_toper, probs
    elif setting == 2:
        new_probs = useYesterdayValue(probs, True)
        return toper, new_probs
    elif setting == 3:
        new_toper = useYesterdayValue(toper, False)
        new_probs = useYesterdayValue(probs, True)
        return new_toper, new_probs
    elif setting == 4:
        new_toper = useAverageValue(toper)
        return new_toper, probs
    elif setting == 5:
        new_probs = useAverageValue(probs)
        return toper, new_probs
    elif setting == 6:
        new_toper = useAverageValue(toper)
        new_probs = useAverageValue(probs)
        return new_toper, new_probs
    elif setting == 7:
        new_toper = randomTopER(len(toper))
        return new_toper, probs
    elif setting == 8:
        new_probs = randomProbs(len(probs))
        return toper, new_probs 
    elif setting == 9:
        new_toper = randomTopER(len(toper))
        new_probs = randomProbs(len(probs))
        return new_toper, new_probs 
    else:
        raise ValueError("Invalid ablation setting")
    
    