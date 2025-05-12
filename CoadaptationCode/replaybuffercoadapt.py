from replaybuffer import EnvReplayBuffer
from rlkit.data_management.replay_buffer import ReplayBuffer
import numpy as np
import torch

class CoadaptReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size_species,
            max_replay_buffer_size_population,
            env,
            env_info_sizes=None
    ):
        self._env = env
        self._max_replay_buffer_size_species = max_replay_buffer_size_species
        self._max_replay_buffer_size_population = max_replay_buffer_size_population

        # default mode 
        self._mode = "species"

        self._ep_counter = 0
        self._expect_init_state = True # LOOK AT THIS VARIABLE?
      
        # init replay buffers
        self._individual_buffer = EnvReplayBuffer(env=self._env, max_replay_buffer_size=self._max_replay_buffer_size_species)
        self._population_buffer = EnvReplayBuffer(env=self._env, max_replay_buffer_size= self._max_replay_buffer_size_population)
        self._init_state_buffer = EnvReplayBuffer(env=self._env, max_replay_buffer_size= self._max_replay_buffer_size_population)
    
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # remove env from state to make it picklable
    #     if 'env' in state:
    #         state['env'] = None
    #     if '_env' in state:
    #         state['_env'] = None
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # reset of env
    #     self.env = None
    #     self._env = None

    def dump(self, filepath: str):
        """Safely save the replay buffer to a file, excluding non-pickleable items."""
        safe_dict = {}

        # only copy safe items
        for k, v in self.__dict__.items():
            if "env" in k or "lock" in k or "socket" in str(type(v)):
                continue
            try:
                torch.save(v, filepath + ".tmp")  # test if it's savable
                safe_dict[k] = v
            except Exception as e:
                print(f"skipping key '{k}' in replay buffer (unsavable): {e}")

        torch.save({'buffer': safe_dict}, filepath)
        print(f"replay buffer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, env=None):
        """Load the replay buffer and reattach the environment."""
        saved = torch.load(filepath)
        buffer = cls.__new__(cls)
        buffer.__dict__.update(saved['buffer'])

        # restore env manually
        buffer.env = env
        buffer._env = env
        print(f"replay buffer loaded from {filepath}")
        return buffer


    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._individual_buffer.add_sample(observation=observation, action=action, reward=reward, terminal=terminal, next_observation=next_observation, **kwargs)
        self._population_buffer.add_sample(observation=observation, action=action, reward=reward, terminal=terminal, next_observation=next_observation, **kwargs)

        # TODO: What is the point of an intitial state replay buffer?
        if self._expect_init_state:
            self._init_state_buffer.add_sample(observation=observation, action=action, reward=reward, terminal=terminal, next_observation=next_observation, **kwargs)
            self._init_state_buffer.terminate_episode() # right now terminate episode is a pass but could change
            self._expect_init_state = False

    def terminate_episode(self):
        """
        :return: # of unique items that can be sampled.
        """
        
        #if self._mode == "species": # double check why we should check this??

        self._individual_buffer.terminate_episode()
        self._population_buffer.terminate_episode()
        self._ep_counter += 1
        self._expect_init_state = True

    def num_steps_can_sample(self, **kwargs):

        if self._mode == "species":
            return self._individual_buffer.num_steps_can_sample(**kwargs)
        elif self._mode == "population":
            return self._population_buffer.num_steps_can_sample(**kwargs)
        else:
            pass

    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        if self._mode == "species":
            # TODO: Figure out what to put here
            ind_batch_size = int(np.floor(batch_size * 0.9))
            pop_batch_size = int(np.ceil(batch_size * 0.1))
            pop = self._population_buffer.random_batch(pop_batch_size)
            spec = self._individual_buffer.random_batch(ind_batch_size)
            for key, item in pop.items():
                pop[key] = np.concatenate([pop[key], spec[key]], axis=0)
            return pop
 
    
        elif self._mode == "population":
            return self._population_buffer.random_batch(batch_size)
        
        elif self._mode == "start":
            return self._init_state_buffer.random_batch(batch_size)
        
        else:
            pass

    
    def set_mode(self, mode):
        if mode == "species": # TODO: change to "individual"
            self._mode = mode
        elif mode == "population":
            self._mode = mode
        elif mode == "start":
            self._mode = mode
        else:
            print("No known mode :(")

    
    def reset_individual_buffer(self):
        self._individual_buffer= EnvReplayBuffer(env=self._env, max_replay_buffer_size=self._max_replay_buffer_size_species)
        self._ep_counter = 0 # reset number of episodes for next design
