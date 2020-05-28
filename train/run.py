import argparse
import gym
import ma_gym
import os
import sys
import pickle
import time
import datetime
import multiprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from utils.replay_memory import Memory
from utils.torch import *
from utils.args import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy

try:
    path = os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name))
    models_file=open(path,'r')
    print("pre-trained models loaded.")
    args.model_path = path
    print("model path: ", path)
except IOError:
    print("pre-trained models not found.")

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space[0].shape[0]
is_disc_action = len(env.action_space[0].shape) == 0

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

p_nets = []
v_nets = []
p_opts = []
v_opts = []

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        for i in range(env.n_agents):
            p_nets.append(DiscretePolicy(args.dec_agents, env.n_agents, state_dim, env.action_space[0].n))
            v_nets.append(Value(env.n_agents, state_dim))
            # add only one policy and value networks if using team unified network settings.
            if args.dec_agents is False:
                break
    else:
        policy_net = Policy(state_dim, env.action_space[0].n, log_std=args.log_std)
else:
    p_nets, v_nets, running_state = pickle.load(open(args.model_path, "rb"))

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')

for i in range(env.n_agents):
    p_nets[i].to(device)
    v_nets[i].to(device)
    if args.dec_agents is False:
        break

state = env.reset()
# state = env.rsi_reset({0:[0,0],1:[0,1],2:[1,0],3:[1,1]})
team_reward = 0

for t in range(10000):
    state_var = tensor(state).unsqueeze(0)
    action = []

    with torch.no_grad():
        for i in range(env.n_agents):
            action += p_nets[i].select_action(state_var)
            if args.dec_agents is False:
                break
    next_state, reward, done, _ = env.step(action)

    team_reward += sum(reward)
    print(team_reward)

    if args.dec_agents is False:
        mask = 0 if all(done) else 1
    else:
        mask = [bool(1-e) for e in done]

    env.render()
    time.sleep(1)

    if all(done):
        break
    
    state = next_state