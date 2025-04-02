import matplotlib.pyplot as plt
from collections import Counter
import re

# Update path for imports
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# For the "Action: #" case
def extract_number(line):
    match = re.search(r'Action:\s*(\d+)', line)
    if match:
        return int(match.group(1))
    elif line.strip().isdigit():
        return int(line.strip())
    return None


# For the "Action: #" case
def count_numbers_modified(filename):
    with open(filename, 'r') as file:
        numbers = [extract_number(line) for line in file]
    numbers = [num for num in numbers if num is not None]
    return Counter(numbers)


# Traditional Case
def count_numbers(filename):
    with open(filename, 'r') as file:
        numbers = [int(line.strip()) for line in file if line.strip().isdigit()]
    return Counter(numbers)


def plot_histogram(counter, number_mapping, out_path, title):
    numbers, counts = zip(*sorted(counter.items()))
    labels = [number_mapping.get(num, str(num)) for num in numbers]
    
    plt.figure(figsize=(12, 10))
    plt.bar(labels, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.title(f'Count of Actions for {title}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(out_path)
    plt.clf()

filename = 'ReinforcementLearning/output/actionHistory/textOut/recon_res_noremoval-437668.out'
number_mapping = {
    0: 'Edge_oo',
    1: 'Edge_nn',
    2: 'Edge_on',
    3: 'Edge_oon',
    4: 'Remove Edge',
    5: 'Add Old Node',
    6: 'Add New Node',
    7: 'Remove Node'
    }  # Define external mapping here
counter = count_numbers(filename)
out_path = 'ReinforcementLearning/output/actionHistory/histograms/recon_res_noresources-437668.png'
title = 'No Resources Testing'
plot_histogram(counter, number_mapping, out_path, title)

# filename = 'ReinforcementLearning/output/actionHistory/textOut/recon_res_base-434689.out'
# number_mapping = {
#     0: 'Edge_oo',
#     1: 'Edge_nn',
#     2: 'Edge_on',
#     3: 'Edge_oon',
#     4: 'Remove Edge',
#     5: 'Add Old Node',
#     6: 'Add New Node',
#     7: 'Remove Node'
#     }  # Define external mapping here
# counter = count_numbers(filename)
# out_path = 'ReinforcementLearning/output/actionHistory/histograms/recon_res_base-434689.png'
# title = 'Base Testing'
# plot_histogram(counter, number_mapping, out_path, title)

'''filename = 'ReinforcementLearning/output/actionHistory/textOut/recon_res_base-434689.out'
number_mapping = {}  # Define external mapping here
counter = count_numbers(filename)
out_path = 'ReinforcementLearning/output/actionHistory/histograms/recon_res_base-434689.png'
title = 'No Removal Allowed Testing'
plot_histogram(counter, number_mapping, out_path, title)'''

# filename = 'ReinforcementLearning/output/actionHistory/textOut/recon_res_grouped-434687.out'
# number_mapping = {
#     0: 'Add Edge',
#     1: 'Remove Edge',
#     2: 'Add Old Node',
#     3: 'Add New Node',
#     4: 'Remove Node'
#     }  # Define external mapping here
# counter = count_numbers_modified(filename)
# out_path = 'ReinforcementLearning/output/actionHistory/histograms/recon_res_grouped-434687.png'
# title = 'Grouped Testing'
# plot_histogram(counter, number_mapping, out_path, title)
