import argparse
import gym
import ma_gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class plot_logger():
    def __init__(self):
        super().__init__()
        self._log = { "Iterations":[], "Min Rewards":[], "Max Rewards":[], "Avg Rewards":[] }
        return

    def log(self, n, r_min, r_max, r_avg):
        self._log["Iterations"].append(n)
        self._log["Min Rewards"].append(r_min)
        self._log["Max Rewards"].append(r_max)
        self._log["Avg Rewards"].append(r_avg)
        return
        