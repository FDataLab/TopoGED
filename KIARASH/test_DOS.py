import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from script.utils.data_util import load_DOS_snapshot_feature

dos_feat = load_DOS_snapshot_feature("IOTX")

print(len(dos_feat))


