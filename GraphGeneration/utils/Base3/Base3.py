"""
CODE TAKEN FROM https://github.com/ekmpa/Base3/blob/main/Base3.py
"""

import numpy as np
from collections import defaultdict, deque


class tCoMem:
    """ 
    Predicts a link based on node co-occurence and the target's
    neighbourhoods' PopTrack popularity within a time window
    """
    def __init__(self, time_window=1_000_000, co_occurrence_weight=0.8):
        self.node_to_recent_dests = defaultdict(lambda: deque(maxlen=100)) 
        self.node_to_co_occurrence = defaultdict(dict)  
        self.time_window = time_window
        self.co_occurrence_weight = co_occurrence_weight

    def update(self, u, v, t):
        self.node_to_recent_dests[u].append((t, v))
        # Co-occurrence as sparse dicts
        self.node_to_co_occurrence[u][v] = self.node_to_co_occurrence[u].get(v, 0) + 1
        self.node_to_co_occurrence[v][u] = self.node_to_co_occurrence[v].get(u, 0) + 1
  
    def get_score(self, u, v, current_time, poptrack_model):
        score = 0.0
        recent = self.node_to_recent_dests.get(u, [])
        valid_recent = [(ts, nbr) for ts, nbr in recent if 0 <= current_time - ts <= self.time_window]
        co_occurrence_score = self.node_to_co_occurrence.get(u, {}).get(v, 0)

        for ts, nbr in valid_recent:
            decay = np.exp(-(current_time - ts) / self.time_window)
            pop_score = poptrack_model.get_score(nbr)
            score += decay * pop_score

        co_occurrence_influence = self.co_occurrence_weight * (co_occurrence_score / (1 + co_occurrence_score))
        score += co_occurrence_influence
        return score / (1 + score) if score > 0 else 0.0
    
  

class PopTrack:
    """
    Implementation follows the paper https://openreview.net/forum?id=9kLDrE5rsW
    """
    def __init__(self, num_nodes, decay=0.95):
        self.decay = decay
        self.popularity = np.zeros(int(num_nodes), dtype=np.float32)  

    def update_batch(self, dest_nodes):
        for dst in dest_nodes:
            dst = int(dst)
            self.popularity[dst] += 1.0
        self.popularity *= self.decay 

    def predict_batch(self, K):
        top_k_indices = np.argsort(self.popularity)[::-1][:K]
        #top_k_scores = self.popularity[top_k_indices]
        return top_k_indices

    def get_score(self, node):
        return self.popularity[int(node)]



# EdgeBank Module
# ==========================
# Code from original github 

def edge_bank_unlimited_memory(sources_list, destinations_list):
    """
    generates the memory of EdgeBank
    The memory stores every edges that it has seen
    """
    # generate memory
    mem_edges = {}
    for e_idx in range(len(sources_list)):
        if (sources_list[e_idx], destinations_list[e_idx]) not in mem_edges:
            mem_edges[(sources_list[e_idx], destinations_list[e_idx])] = 1
    return mem_edges

def edge_bank_infin_freq(sources_list, destinations_list):
    """
    generates the memory of EdgeBank_inf with frequency scores not just 
    The memory stores every edges that it has seen
    """
    # generate memory
    mem_edges = {}
    for e_idx in range(len(sources_list)):
        if (sources_list[e_idx], destinations_list[e_idx]) not in mem_edges:
            mem_edges[(sources_list[e_idx], destinations_list[e_idx])] = 1
        else: 
            mem_edges[(sources_list[e_idx], destinations_list[e_idx])] += 1
    # normalize to 0-1 scale
    if mem_edges:
        max_freq = max(mem_edges.values())
        if max_freq > 0:
            for edge in mem_edges:
                mem_edges[edge] /= max_freq
    return mem_edges

def edge_bank_repetition_based_memory(sources_list, destinations_list):
    """
    in memory, save edges that has repeated more than a threshold
    """
    # frequency of seeing each edge
    all_seen_edges = {}
    for e_idx in range(len(sources_list)):
        if (sources_list[e_idx], destinations_list[e_idx]) in all_seen_edges:
            all_seen_edges[(sources_list[e_idx], destinations_list[e_idx])] += 1
        else:
            all_seen_edges[(sources_list[e_idx], destinations_list[e_idx])] = 1
    n_repeat = np.array(list(all_seen_edges.values()))
    # NOTE: different values can be set to the threshold with manipulating the repeat_occur dictionary
    threshold = np.mean(n_repeat)
    mem_edges = {}
    for edge, n_e_repeat in all_seen_edges.items():
        if n_e_repeat >= threshold:
            mem_edges[edge] = 1
    return mem_edges

def time_window_edge_memory(sources_list, destinations_list, timestamps_list, start_time, end_time):
    """
    returns a memory that contains all edges seen during a time window
    """
    mem_mask = np.logical_and(timestamps_list <= end_time, timestamps_list >= start_time)
    src_in_window = sources_list[mem_mask]
    dst_in_window = destinations_list[mem_mask]
    mem_edges = edge_bank_unlimited_memory(src_in_window, dst_in_window)
    return mem_edges

def edge_bank_time_window_memory(sources_list, destinations_list, timestamps_list, window_mode, memory_span=0.01):
    window_end_ts = timestamps_list.max()

    """
    only saves the edges seen the time time interval equal to the last time window in timestamps_list
    """
    if window_mode == 'fixed':
        window_start_ts = np.quantile(timestamps_list, 1 - memory_span)
        window_end_ts = max(timestamps_list)
    elif window_mode == 'avg_reoccur':
        e_ts_l = {}
        for e_idx in range(len(sources_list)):
            curr_edge = (sources_list[e_idx], destinations_list[e_idx])
            if curr_edge not in e_ts_l:
                e_ts_l[curr_edge] = []
            e_ts_l[curr_edge].append(timestamps_list[e_idx])

        sum_t_interval = 0
        for e, ts_list in e_ts_l.items():
            if len(ts_list) > 1:
                ts_interval_l = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
                sum_t_interval += np.mean(ts_interval_l)
        avg_t_interval = sum_t_interval / len(e_ts_l)
        window_end_ts = max(timestamps_list)
        window_start_ts = window_end_ts - avg_t_interval
        
    mem_edges = time_window_edge_memory(sources_list, destinations_list, timestamps_list, start_time=window_start_ts,
                                        end_time=window_end_ts)
    return mem_edges

def edgebank_score(u, v, edgebank):
    return 1.0 if (u, v) in edgebank else 0.0

def edgebank_freq_score(u, v, edgebank):
    return edgebank.get((u, v), 0)


# Weighting functions
# -------------------
# The weighting combniation can either weight each component equally (uniform),
# or be based on combinations found empirically to work well (not used anymore), or based on a confidence signal
# regarding the EdgeBank and PopTrack scores. 

def weights_EB_signal(eb_conf):
    """ 
    eb_conf is eb_score. the idea is that if the score is 1,
    the link is in the edgebank, so the edgebank score is more 
    meaningful than if not. 
    """

    # Define weights when confident in EdgeBank vs not
    weights_if_confident = np.array([0.5, 0.2, 0.3]) 
    weights_if_not = np.array([0.2, 0.3, 0.5])

    # Blend based on EdgeBank confidence
    blended_weights = eb_conf * weights_if_confident + (1 - eb_conf) * weights_if_not
    blended_weights /= blended_weights.sum()  # normalize 

    return blended_weights

def weights_manual(eb_conf): 
    """
    Based on weights found through manual tuning
    Logic still based on edgebank confidence signal
    """
    if eb_conf:
        alpha, beta, delta = 0.5, 0.2, 0.3 
    else: 
        alpha, beta, delta = 0.2, 0.3, 0.5
    return alpha, beta, delta

def weights_uniform():
    """
    Uniform weights for all components (1/3)
    """
    return 0.33, 0.33, 0.33


def multi_conf(eb_score, pop_score):
    """
    Compute interpolation weights for the Base3 model based on confidence signals 
    from EdgeBank (eb_score) and PopTrack (pop_score).

    Interpretation of cases:
        - Both EdgeBank and PopTrack hit (high confidence): assign higher weight to PopTrack (0.45) and moderate to EdgeBank (0.35)
        - Only EdgeBank hits: give more emphasis to EdgeBank (0.45) with less reliance on PopTrack (0.25)
        - Only PopTrack hits: PopTrack dominates (0.7) since EdgeBank is inapplicable
        - Neither hits: fallback to more balanced reliance, favoring PopTrack slightly (0.45 vs. 0.20)

    The final weight for t-CoMem (w_t) is always set to ensure the weights sum to 1.
    """
    
    if eb_score: 
        if pop_score: 
            w_eb, w_pt = 0.35, 0.45
        else: 
            w_eb, w_pt = 0.45, 0.25 
        w_t = 1 - (w_eb + w_pt)
    else:
        if pop_score: 
            w_eb, w_pt = 0.15, 0.7
        else: 
            w_eb, w_pt = 0.20, 0.45
        w_t = 1 - (w_eb + w_pt)

    return w_eb, w_pt, w_t
    


# Actual scoring function called by model
def full_interpolated_score(u, v, t, edgebank, poptrack_model, top_k_nodes, coMem, method='multi_conf'):
    """
    Interpolated score with weights based on method
    """
    eb_score = edgebank_score(u, v, edgebank)
    pop_score = 1.0 if v in top_k_nodes else 0.0
    co_score = coMem.get_score(u, v, t, poptrack_model)

    if method == 'EB_conf':
        alpha, beta, delta = weights_EB_signal(eb_score)
    elif method == 'uniform':
        alpha, beta, delta = weights_uniform()
    elif method == 'manual':
        alpha, beta, delta = weights_manual(eb_score) 
    elif method == 'EdgeBank':
        alpha, beta, delta = 1, 0, 0
    elif method == 'PopTrack':
        alpha, beta, delta = 0, 1, 0
    elif method == 'tCoMem':
        alpha, beta, delta = 0, 0, 1
    elif method == 'multi_conf':
        alpha, beta, delta = multi_conf(eb_score, pop_score)
    else:
        alpha, beta, delta = weights_uniform()

    return alpha * eb_score + beta * pop_score + delta * co_score