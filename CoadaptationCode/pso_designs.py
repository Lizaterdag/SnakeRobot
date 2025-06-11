import os
import torch
from pso_batch import PSO_batch
from snakeenv_thread_coadapt import SnakeEnv

from replaybuffercoadapt import CoadaptReplayBuffer
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import rlkit.torch.networks 

def identity(x):
    return x

# some monkey patching to make sure rlkit works with the replay buffer
rlkit.torch.networks.identity = identity


REPLAY_PATHS = [
    "replay/replay_2025_06_03_Design0_carpet.pt",
    "replay/replay_2025_06_02_Design2_carton.pt",
    "replay/replay_2025_06_02_Design0_foam.pt",
    "replay/replay_2025_06_02_Design0_carton.pt",
    "replay/replay_2025_05_30_Design1_carton.pt",
    "replay/replay_2025_05_30_Design1_carpet.pt",
    "replay/replay_2025_05_26_Design2_foam.pt",
    "replay/replay_2025_05_26_Design1_carton.pt",

]

env = SnakeEnv()

# new population replay buffer
pop_replay = CoadaptReplayBuffer(
    max_replay_buffer_size_species=int(1e6),
    max_replay_buffer_size_population=int(1e7),
    env=env,
    env_info_sizes=None
)

print("Loading selected episodes into population buffer...")
for path in REPLAY_PATHS:
    data = torch.load(path)
    num_samples = data['_size']
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    next_obs = data['next_observations']
    terminals = data['terminals']

    
    episode_length = 176  
    for ep in range(17, 31):  # episodes 17 to 30
        start_idx = ep * episode_length
        end_idx = min((ep + 1) * episode_length, num_samples)
        for i in range(start_idx, end_idx):
            pop_replay.add_sample(
                observation=observations[i],
                action=actions[i],
                reward=rewards[i],
                next_observation=next_obs[i],
                terminal=terminals[i],
                env_info={}
            )
print("Done loading selected episodes.")


q_network = torch.load("results/ind_qf1_tar_2025_05_12_Design1_ep2.pt", map_location=torch.device('cpu'))
policy_network = torch.load("results/ind_policy_2025_05_12_Design1_ep2.pt", map_location=torch.device('cpu'))


pso = PSO_batch(pop_replay, env)

# random starting design
init_design = [45.0, 5, 50.0, 0.0] 

# run design optimization
print("running PSO")
cost, best_design = pso.optimize_design(
    design=init_design,
    q_network=q_network,
    policy_network=policy_network
)

#clip material parameter to [0, 1]
best_design[-1] = np.round(np.clip(best_design[-1], 0, 1))

print("best morphology parameters:", best_design)
print("cost:", cost)