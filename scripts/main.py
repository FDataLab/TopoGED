import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import Utils

my_utils = Utils()
path = 'data/output/results/RegressionTesting/data/'
for dataset in ["CollegeMsg", "networkaeternity", "networkaion", "networkaragon"]:
    my_utils.distances_from_overshoots(path + dataset + '_overshoots.csv')