import rlkit.torch.pytorch_util as ptu

class RLAlgorithm(object):

    def __init__(self, env, replay, networks):
        # self._config = config
        # self.file_str = config['data_folder_experiment']

        ptu.set_gpu_mode(True)
        self._env = env
        self._networks = networks
        self._replay = replay

        ptu.set_gpu_mode(True) # added this

        # if 'use_only_global_networks' in config.keys():
        #     self._use_only_global_networks = config['use_only_global_networks']
        # else:
        #     self._use_only_global_networks = False

    def episode_init(self):
        pass