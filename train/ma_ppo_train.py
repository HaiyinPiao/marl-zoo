import argparse
import gym
import ma_gym
import os
import sys
import pickle
import time
import datetime
import copy
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from utils.args import *
from plot.plot_logger import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

try:
    path = os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name))
    models_file=open(path,'r')
    print("pre-trained models loaded.")
    args.model_path = path
    print("model path: ", path)
except IOError:
    print("pre-trained models not found.")

if args.log_plot is True:
    plotlogger = plot_logger()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)



"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space[0].shape[0]
is_disc_action = len(env.action_space[0].shape) == 0
# running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)
running_state = None

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

for i in range(env.n_agents):
    p_nets[i].to(device)
    v_nets[i].to(device)
    p_opts.append(torch.optim.Adam(p_nets[i].parameters(), lr=args.learning_rate))
    v_opts.append(torch.optim.Adam(v_nets[i].parameters(), lr=args.learning_rate))
    if args.dec_agents is False:
        break

"""create agent"""
agent = Agent(env, p_nets, device, running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)

    values = []
    fixed_log_probs = []
    with torch.no_grad():
        if args.dec_agents is False:
            values = v_nets[0](states)
            fixed_log_probs = p_nets[0].get_log_prob(states, actions)
        else:
            for i in range(env.n_agents):
                values.append(v_nets[i](states))
                fixed_log_probs.append(p_nets[i].get_agent_i_log_prob(i, states, actions))
            values = torch.stack(values)
            values = torch.transpose(values,0,1)
            fixed_log_probs = torch.stack(fixed_log_probs)
            fixed_log_probs = torch.transpose(fixed_log_probs,0,1)

    """get advantage estimation from the trajectories"""
    advantages = []
    returns = []
    if args.dec_agents is False:
        rewards_sum = torch.sum(rewards, dim=1)
        advantages, returns = estimate_advantages(rewards_sum, masks, values, args.gamma, args.tau, device)
    else:
        for i in range(env.n_agents):
            adv, ret = estimate_advantages(rewards[:,i], masks[:,i], values[:,i,:], args.gamma, args.tau, device)
            advantages.append(adv)
            returns.append(ret)
        advantages = torch.stack(advantages)
        advantages = torch.transpose(advantages,0,1)
        returns = torch.stack(returns)
        returns = torch.transpose(returns,0,1)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            if args.dec_agents is False:
                ppo_step(p_nets[0], v_nets[0], p_opts[0], v_opts[0], 5, states_b, actions_b, returns_b,
                        advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)
            else:
                for i in range(env.n_agents):
                    ppo_step(p_nets[i], v_nets[i], p_opts[i], v_opts[i], 5, states_b, actions_b, returns_b[:,i],
                            advantages_b[:,i], fixed_log_probs_b[:,i], args.clip_epsilon, args.l2_reg, i)                    


def main_loop():
    # RSI randomization from previous sampling replay memory
    rsi_mem_prev = None

    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, rsi_mem_prev)

        if args.rsi is True:
            rsi_mem_prev = copy.copy(batch)

        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))
            if args.log_plot is True:
                plotlogger.log(n=i_iter, r_min=log['min_reward'], r_max=log['max_reward'], r_avg=log['avg_reward'])

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            for i in range(env.n_agents):
                to_device(torch.device('cpu'), p_nets[i], v_nets[i])
                if args.dec_agents is False:
                    break
            
            print("logging trained models.")
            pickle.dump((p_nets, v_nets, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))

            for i in range(env.n_agents):
                p_nets[i].to(device)
                v_nets[i].to(device)
                if args.dec_agents is False:
                    break

        if args.log_plot is True and i_iter%args.log_plot_steps==0 and i_iter>=args.log_plot_steps:
            logplot_path = os.path.join(assets_dir(), 'learned_models/')
            with open(os.path.join(logplot_path+"logplot"+str(datetime.datetime.now())+".pkl"), "wb") as f: pickle.dump(plotlogger._log, f, pickle.HIGHEST_PROTOCOL)
            print("plot log succeed.")
            args.log_plot = False
            exit()

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
