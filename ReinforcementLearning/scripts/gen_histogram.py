# ''import matplotlib.pyplot as plt
# from collections import Counter
# import re

# # Update path for imports
# import os
# import sys
# import argparse
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# # For the "Action: #" case
# def extract_number(line):
#     match = re.search(r'Action:\s*(\d+)', line)
#     if match:
#         return int(match.group(1))
#     elif line.strip().isdigit():
#         return int(line.strip())
#     return None


# # For the "Action: #" case
# def count_numbers_modified(filename):
#     with open(filename, 'r') as file:
#         numbers = [extract_number(line) for line in file]
#     numbers = [num for num in numbers if num is not None]
#     return Counter(numbers)

# # def count_numbers(filename):
# #     counter = Counter({i: 0 for i in range(8)})  # Initialize counts for 0-7

# #     with open(filename, 'r', encoding='utf-16') as file:
# #         for line in file:
# #             stripped = line.strip().replace("\u202c", "").replace("\ufeff", "")  # Remove hidden Unicode characters
# #             if stripped.isdigit() and int(stripped) in range(8):  # Check for valid numbers
# #                 num = int(stripped)
# #                 counter[num] += 1  # Increment count

# #     return counter


# # Traditional Case
# def count_numbers(filename):
#     with open(filename, 'r') as file:
#         numbers = [int(line.strip()) for line in file if line.strip().isdigit()]
#     return Counter(numbers)


# def plot_histogram(counter, number_mapping, out_path, title):
#     numbers, counts = zip(*sorted(counter.items()))
#     labels = [number_mapping.get(num, str(num)) for num in numbers]
    
#     plt.figure(figsize=(12, 10))
#     plt.bar(labels, counts, color='skyblue', edgecolor='black')
#     plt.xlabel('Action')
#     plt.ylabel('Frequency')
#     plt.title(f'Count of Actions for {title}')
#     plt.xticks(rotation=45, ha='right')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     plt.savefig(out_path)
#     plt.clf()

# filename = 'ReinforcementLearning/output/actionHistory/textOut/testing_out_noresources_imitation.out'
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
# print(counter)
# out_path = 'ReinforcementLearning/output/actionHistory/histograms/testing_out_noresources_imitation.png'
# title = 'No Resources Testing Imitation'
# plot_histogram(counter, number_mapping, out_path, title)

# # filename = 'ReinforcementLearning/output/actionHistory/textOut/recon_res_base-434689.out'
# # number_mapping = {
# #     0: 'Edge_oo',
# #     1: 'Edge_nn',
# #     2: 'Edge_on',
# #     3: 'Edge_oon',
# #     4: 'Remove Edge',
# #     5: 'Add Old Node',
# #     6: 'Add New Node',
# #     7: 'Remove Node'
# #     }  # Define external mapping here
# # counter = count_numbers(filename)
# # out_path = 'ReinforcementLearning/output/actionHistory/histograms/recon_res_base-434689.png'
# # title = 'Base Testing'
# # plot_histogram(counter, number_mapping, out_path, title)

# '''filename = 'ReinforcementLearning/output/actionHistory/textOut/recon_res_base-434689.out'
# number_mapping = {}  # Define external mapping here
# counter = count_numbers(filename)
# out_path = 'ReinforcementLearning/output/actionHistory/histograms/recon_res_base-434689.png'
# title = 'No Removal Allowed Testing'
# plot_histogram(counter, number_mapping, out_path, title)'''

# # filename = 'ReinforcementLearning/output/actionHistory/textOut/recon_res_grouped-434687.out'
# # number_mapping = {
# #     0: 'Add Edge',
# #     1: 'Remove Edge',
# #     2: 'Add Old Node',
# #     3: 'Add New Node',
# #     4: 'Remove Node'
# #     }  # Define external mapping here
# # counter = count_numbers_modified(filename)
# # out_path = 'ReinforcementLearning/output/actionHistory/histograms/recon_res_grouped-434687.png'
# # title = 'Grouped Testing'
# # plot_histogram(counter, number_mapping, out_path, title)
''

import re
from collections import Counter
import matplotlib.pyplot as plt

def parse_out_file(file_path):
    number_counts = Counter()
    keyword_counts = Counter({'TIMEOUT': 0, 'success': 0})

    with open(file_path, 'r', encoding='utf-16') as f:
        for line in f:
            stripped = line.strip()

            # Count whole-number lines
            if re.fullmatch(r'\d+', stripped):
                number = int(stripped)
                number_counts[number] += 1

            # Count keyword appearances
            if 'TIMEOUT' in stripped:
                keyword_counts['TIMEOUT'] += 1
            if 'success' in stripped.lower():
                keyword_counts['success'] += 1

    return number_counts, keyword_counts

def plot_histogram(number_counts, keyword_counts, out_path):
    labels = list(map(str, number_counts.keys())) + ['TIMEOUT', 'success']
    values = list(number_counts.values()) + [keyword_counts['TIMEOUT'], keyword_counts['success']]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='skyblue')
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.title('Histogram of Actions, TIMEOUT, and Success Occurrences (No Removal)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(out_path)

if __name__ == '__main__':
    number_mapping = {
        0: 'Edge_oo',
        1: 'Edge_nn',
        2: 'Edge_on',
        3: 'Edge_oon',
        4: 'Add Old Node',
        5: 'Add New Node',
        }

    file_path = 'ReinforcementLearning/output/actionHistory/textOut/nx_noremoval copy.out'  # Replace this with your actual filename
    out_path = 'ReinforcementLearning/output/actionHistory/histograms/nx_noremoval.png'
    # with open(file_path, 'r', encoding='utf-16') as f:
    #     for i, line in enumerate(f):
    #         print(f"Line {i+1!r}: {line!r}")
    number_counts, keyword_counts = parse_out_file(file_path)
    plot_histogram(number_counts, keyword_counts, out_path)