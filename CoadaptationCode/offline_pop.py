import torch
from SACTrainer import SACTrainer
from soft_actor_critic_coadapt import SoftActorCriticCoadapt
from replaybuffercoadapt import CoadaptReplayBuffer
from snakeenv_thread_coadapt import SnakeEnv
import rlkit.torch.pytorch_util as ptu
import numpy as np
from gymnasium import spaces


class WrappedSnakeEnv(SnakeEnv):
    def __init__(self, design_dim=4):
        super().__init__()
        base_dim = self.observation_space.shape[0]
        self._design_dim = design_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )

ptu.set_gpu_mode(False)

REPLAY_PATHS = [
    "replay/replay_2025_06_03_Design0_carpet.pt",
    "replay/replay_2025_06_02_Design2_carton.pt",
    "replay/replay_2025_06_02_Design0_foam.pt",
    "replay/replay_2025_06_02_Design0_carton.pt",
    "replay/replay_2025_05_30_Design1_carton.pt",
    "replay/replay_2025_05_30_Design1_carpet.pt",
    "replay/replay_2025_05_26_Design2_foam.pt",
    "replay/replay_2025_05_26_Design1_carton.pt",
    "replay/replay_2025_05_26_Design1_carton.pt",
    "replay/replay_2025_06_18_Design4_carpet.pt",
    "replay/replay_2025_06_17_Design4_carton.pt",
    "replay/replay_2025_06_12_Design4_foam.pt",
    "replay/replay_2025_06_16_Design5_carton.pt",
    "replay/replay_2025_06_18_Design5_carpet.pt",
    "replay/replay_2025_06_16_Design5_foam.pt",

]

env = WrappedSnakeEnv(design_dim=4)

networks = SoftActorCriticCoadapt.create_networks(env)

pop_replay = CoadaptReplayBuffer(
    max_replay_buffer_size_species=int(1e6),
    max_replay_buffer_size_population=int(1e7),
    env=env,
    env_info_sizes=None
)

episode_length = 175
design_dim = 4

for path in REPLAY_PATHS:
    data = torch.load(path)
    num_samples = data['_size']
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    next_obs = data['next_observations']
    terminals = data['terminals']

    for ep in range(17, 31):
        start = ep * episode_length
        end = min((ep + 1) * episode_length, num_samples)
        for i in range(start, end):
            pop_replay.add_sample(
                observation=observations[i],          
                action=actions[i],
                reward=rewards[i],
                next_observation=next_obs[i],         
                terminal=terminals[i],
                env_info={}
            )

trainer = SoftActorCriticCoadapt(
    env=env,
    replay=pop_replay,
    networks=networks,
)

trainer._replay.set_mode("population")
trainer._nmbr_pop_updates = 500

max_epochs = 500



for epoch in range(max_epochs):
    print(f"\nEpoch {epoch + 1}")
    _, _, _, popQ1, popQ2, popPol = trainer.single_train_step(train_ind=False, train_pop=True)

    # total_loss = abs(popQ1[0]) + abs(popQ2[0]) + abs(popPol[0])
    print(f"Policy loss: {abs(popPol[0])}")

#save models
pop_policy = networks["population"]["policy"]
pop_qf1 = networks["population"]["qf1"]

torch.save(pop_policy.state_dict(), "pop_policy.pt")
torch.save(pop_qf1.state_dict(), "pop_qf1_epoch.pt")

print("models saved")
