import argparse
import torch

# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


parser = argparse.ArgumentParser(description='Args to run script')
# 1.dataset
parser.add_argument('--dataset', type=str, default='enron10', help='datasets')


# 2.experiments
parser.add_argument('--test_ratio', type=int, default=0.15, help='ratio for testing, default:0.15')
parser.add_argument('--val_ratio', type=int, default=0.15, help='ratio for evaluation, default:0.15')

args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda".format(args.device_id))
    print('INFO: using gpu:{} to train the models'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('INFO: using cpu to train the models')
