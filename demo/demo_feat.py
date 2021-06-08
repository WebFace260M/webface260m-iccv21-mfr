import sys
import os
import datetime
import numpy as np
import cv2
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

parser = argparse.ArgumentParser(description='Run MFR online validation.')
parser.add_argument('--path', type=str, default='../',
                    help='mfr implementation path')
args = parser.parse_args()
_path = args.path

sys.path.append(_path)
from pywebface260mmfr_implement import PyWebFace260M

x = PyWebFace260M()
assets_path = os.path.join(_path, 'assets')
x.load(assets_path)
feat_len = x.feat_dim
print('feat length:', feat_len)

feat_list = []
for i in range(6):
    img_path = "{}.png".format(i)
    ta = datetime.datetime.now()
    img = cv2.imread(img_path)
    feat = x.get_feature(img)
    tb = datetime.datetime.now()
    print('cost:', (tb - ta).total_seconds())
    feat_list.append(feat)

feat1 = feat_list[0]
feat2 = feat_list[1]

sim = x.get_sim(feat1, feat2)

print("sim: ", sim)

