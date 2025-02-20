import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
import pyswarms as ps
from design_optimization import Design_Optimization
from snakeenv_thread_coadapt import SnakeEnv
#from snakeenv_thread_coadapt import SnakeEnv

class PSO_batch(Design_Optimization):

    def __init__(self, replay, env):
        self._replay = replay
        self._env = env

        self._state_batch_size = 32

    def optimize_design(self, design, q_network, policy_network):
        self._replay.set_mode('start')

        initial_state = self._replay.random_batch(self._state_batch_size)
        initial_state = initial_state['observations']
        design_idx = SnakeEnv.get_design_dimensions()

        def f_qval(x_input, **kwargs): # function to optimize
            shape = x_input.shape
            print('SHAPE DONE')
            cost = np.zeros((shape[0],))

            with torch.no_grad():
                for i in range(shape[0]):
                    x = x_input[i:i+1,:]
                    state_batch = initial_state.copy()
                    state_batch[:,design_idx] = x
                    network_input = torch.from_numpy(state_batch).to(device=ptu.device, dtype=torch.float32)
                    action,_, _, _, _, _, _, _, = policy_network(network_input, deterministic=True)
                    output = q_network(network_input, action)
                    print(output)
                    loss = -output.mean().sum()
                    fval = float(loss.item())
                    cost[i] = fval
                    
            return cost

        # TODO: MODIFIED FROM ORIGINAL CHECK THAT WORKS
        lower_bounds = [] 
        upper_bounds = []


        lower_bounds = [l for l, _ in SnakeEnv.design_parameter_bounds]
        lower_bounds = np.array(lower_bounds)
        upper_bounds = [u for _, u in SnakeEnv.design_parameter_bounds]
        upper_bounds = np.array(upper_bounds)
        bounds = (lower_bounds, upper_bounds)

       
        # c1 = cognitive parameter
        # c2 = social parameter
        # w = inertia parameter
        # https://pyswarms.readthedocs.io/en/latest/api/pyswarms.single.html
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

        optimizer = ps.single.GlobalBestPSO(n_particles=700, dimensions=len(design), bounds=bounds, options=options)
        
        # Perform optimization
        cost, new_design = optimizer.optimize(f_qval, print_step=100, iters=5, verbose=3) #, n_processes=2) # iter was 250
        print('OPTIMIZED')
        return cost, new_design
