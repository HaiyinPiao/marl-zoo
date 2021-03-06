import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
from utils.args import *
import math
import time
import random


def collect_samples(pid, queue, env, p_nets, custom_reward,
                    mean_action, render, running_state, min_batch_size, rsi_mem_prev=None):
    torch.randn(pid)
    log = dict()
    memory = Memory()
    team_reward = 0.0
    if args.dec_agents is True:
        reward_episodes = [0.0]*env.n_agents
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    while num_steps < min_batch_size:
        if args.rsi is True and rsi_mem_prev is not None:
            # randomized starting point.
            sp = rsi_mem_prev.rsi_state
            rs = random.sample(sp, 1)
            rs = rs[0]
            # print(rs)
            # exit()
            state = env.rsi_reset({0:rs[0],1:rs[1],2:rs[2],3:rs[3]})
        else:
            state = env.reset()

        if running_state is not None:
            state = running_state(state)
        team_reward = 0
        if args.dec_agents is True:
            reward_episodes = [0.0]*env.n_agents

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0)
            action = []

            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    for i in range(env.n_agents):
                        action += p_nets[i].select_action(state_var)
                        if args.dec_agents is False:
                            break
            next_state, reward, done, _ = env.step(action)
            team_reward += sum(reward)
            if args.dec_agents is True:
                reward_episodes += reward
                reward_episodes = [i + j for i, j in zip(reward_episodes, reward)]

            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            if args.dec_agents is False:
                mask = 0 if all(done) else 1
            else:
                mask = [bool(1-e) for e in done]

            if args.rsi is True:
                memory.push(state, action, mask, next_state, reward, env.agent_pos)
            else:
                memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()

            if all(done):
                break
            
            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += team_reward
        min_reward = min(min_reward, team_reward)
        max_reward = max(max_reward, team_reward)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, p_nets, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.p_nets = p_nets
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size, rsi_mem_prev=None):
        t_start = time.time()
        for i in range(self.env.n_agents):
            to_device(torch.device('cpu'), self.p_nets[i])
            if args.dec_agents is False:
                break
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.p_nets, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size, rsi_mem_prev)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.p_nets, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size, rsi_mem_prev)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        for i in range(self.env.n_agents):
            to_device(self.device, self.p_nets[i])
            if args.dec_agents is False:
                break
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
