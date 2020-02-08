# For creating paper style performance comparation figures
# Curve Plotting by using seaborn and matplotlib
# haiyinpiao@qq.com
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import seaborn as sns
from glob import glob
from enum import Enum
from enum import IntEnum
from utils import *
from plot_logger import plot_logger

logplot_path = os.path.join(assets_dir(), 'learned_models/')
log_names = glob(logplot_path+'logplot*.pkl')

iterations = []
r_mins = []
r_maxs = []
r_avgs = []

for name in log_names:
    with open(os.path.join(name), "rb") as f:
        logplot = pickle.load(f)
        iterations.append(logplot["Iterations"])
        r_mins.append(logplot["Min Rewards"])
        r_maxs.append(logplot["Max Rewards"])
        r_avgs.append(logplot["Avg Rewards"])

linestyle = ['-', '--', ':', '-.', '-']
# color = ["#3778bf", "#feb308", "red", "green", "#ff7b00"]
# color = ["#DC5712", "#E58308", "#F4D000", "#8A977B","#B6C29A"]
# color = ["#eb5a1a","#f69a76", "#004d65", "#4e8599", "#a7c4ce"]
# color = ["#ce4e4e","#9fadde", "#536ac1", "#feca12", "#02944d"]
color = ["#ff3d00","#4caf50", "#1976d2", "#ffc107", "#ff7b00"]

plt.figure(figsize=(6.4, 4.8))
sns.set(font_scale=10.0)
sns.set(style="whitegrid")

sns.tsplot(time=logplot["Iterations"], data=r_mins, color=color[0], linestyle=linestyle[0], condition="Min Rewards")
sns.tsplot(time=logplot["Iterations"], data=r_maxs, color=color[1], linestyle=linestyle[0], condition="Max Rewards")
sns.tsplot(time=logplot["Iterations"], data=r_avgs, color=color[2], linestyle=linestyle[0], condition="Avg Rewards")
# plt.xlim((0, 50))
# plt.ylim((-300, 150))
# ax.set_xticks(range(0, 251, 10))
plt.ylabel("Rewards")
plt.xlabel("Iterations")
plt.tight_layout()

plt.show()