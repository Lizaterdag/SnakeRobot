# read motors
import gymnasium as gym
import matplotlib.pyplot as plt
from soft_actor_critic_coadapt import SoftActorCriticCoadapt
from snakeenv_thread_coadapt import SnakeEnv
import numpy as np
from replaybuffercoadapt import CoadaptReplayBuffer
import os
import torch
from scipy.interpolate import interp1d
import threading
import matplotlib.pyplot as plt
import pandas as pd
import os
import utils
import pickle
import gc
from pso_batch import PSO_batch
import time
import rlkit.torch.pytorch_util as ptu
from motorssynced import MotorsSynced

from datetime import datetime

class Train():
    def __init__(self):        

        self.env = gym.make("SnakeRobot")
    
        self._reward_scale = 1.0
        self._episode_length = 300 # number of timesteps per episode
        self.episode_counter = 0
        self.episodes_before_training = 10 # number of episodes before training to fill the replay buffer
        self.episode_iterations = 30 # number of episodes per design
        self.design_cylces = 20 # total number of design cycles

        self.episodeCumulativeRewards = []  # Stores cumulative rewards per episode
        self.cumulativeRewards = []  # Stores cumulative rewards per step

        self.episodeCumulativeRewards = []

        self.eachEpisodeCumuRewards = []

        self.num_init_designs = 5 # number of initial design cycles
        # set up replay
        self.replay = CoadaptReplayBuffer(
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7),
            env= self.env,
            env_info_sizes=None
        )

        # set up RL algorithm
        self.rl_method = SoftActorCriticCoadapt
        self.networks = self.rl_method.create_networks(env=self.env)
        self.rl_alg = self.rl_method(env=self.env, replay=self.replay, networks=self.networks)

        # set up design variables
        self.do_alg = PSO_batch(self.replay, self.env)
        self.design_counter = 2
        self.episode_counter = 0
        self.data_design_type = 'Initial'
        

        self.date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S") # for files
        
        

    def run(self, stopEvent):
        """ Runs the Fast Evolution through Actor-Critic RL algorithm.

        First the initial design loop is executed in which the rl-algorithm
        is exeuted on the initial designs. Then the design-optimization
        process starts.
        It is possible to have different numbers of iterations for initial
        designs and the design optimization process.
        """
        self.stateList = [[] for i in range(0,17)]
        self.actionList = [[] for i in range(0,6)]
        self.designList = [[] for i in range(0,7)]
        self.timestepRewards = []
        self.episodeCumulativeRewards = []
        self.cumulativeRewards = []
        self.epList = []
        self.timesteps = []
        self.epListLoss = []
        self.q1loss = []
        self.q2loss = []
        self.policyloss = []
        self.popq1loss = []
        self.popq2loss = []
        self.poppolicyloss = []

        # setting up files and file names
        self.date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        name = "Rewards_Design{}".format(str(self.design_counter))
        self.filename = self.date+name
        name = "Losses_Design{}".format(str(self.design_counter))
        self.lossFilename = self.date+name

        ptu.set_gpu_mode(True) # making sure to use GPU
       
        #self.optimized_params = [3.5904986579519607, 1.7847799600014809, 3.584383159576394, 3.3680278532796404, 1.2309833188420878, 0.5229256834552103, 3.250148806721444]
        
        # determine what to do based on design cycle currently on
        if self.design_counter < self.num_init_designs: # if not done with initial design loop
            self.initial_design_loop() # run another initial design loop
            print(f'design counter at {self.design_counter}')
            if self.design_counter == self.num_init_designs: # if on last initial design cycle
                # last initial design
                self.first_train_op() # run design
            stopEvent.set() # end thread
            return
        
        
        elif self.design_counter < self.design_cylces: # if still on all other designs
            self.train_loop()
            stopEvent.set() # end thread
            return

        else: # have reached end of training
            # can run end sequence here
            pass 
    

    def collect_training_experience(self):
            """ Collect training data.

            This function executes a single episode in the environment using the
            exploration strategy/mechanism and the policy.
            The data, i.e. state-action-reward-nextState, is stored in the replay
            buffer.

            """

            # reset environment
            state, info = self.env.reset()
            steps = 0
            episodeRewards = 0
            episodeContRewards = []
            Done = False
        
            # get policies
            self.policy = self.rl_alg.get_policy_network(self.networks['individual']) #get policy here
            self.pop_policy = self.rl_alg.get_policy_network(self.networks['population']) #get policy here

            currDesign = SnakeEnv.get_current_design()
            
            while not (Done) and steps <= self._episode_length:
                start = time.time()
                
                self.timesteps.append(steps)


                steps += 1
                print(f'Step: {steps}')
                #state = torch.tensor(state)
                #state = state.to(torch.float32)

                
                # exploration vs exploitation
                if self.currEp > self.episodes_before_training : # can start training, exploitation
                    action,_ = self.policy.get_action(state, deterministic=False) 
                else: # purely exploring
                    #action, _= self.pop_policy.get_action(state, deterministic=False)
                    action = np.random.uniform(-1,1, size=6) # this is for early designs


                for i in range(0,6): # for data logging purpose
                    self.actionList[i].append(action[i])
                
        
                next_state, reward, terminated, truncated, info = self.env.step(action) # step the action, note: reward is scaled in environment

                if steps > self._episode_length:
                    SnakeEnv.disableMotorTorque() # stop motors when reach end of an episode
                    print('disabled torque')
                
                episodeRewards += reward # accumulate rewards here to track for comparison
        
                # log rewards
                self.timestepRewards.append(reward)
                self.cumulativeRewards.append(episodeRewards)
                self.epList.append(self.currEp) # to make note of what episode we are on
                for i in range(0,17):
                    self.stateList[i].append(state[i])


             

                Done = terminated # can check here for terminated and truncated
                terminal = np.array([Done]) # turn into array for replay buffer
                reward = np.array([reward])
                
                # add replay sample
                self.replay.add_sample(observation=state, action=action, reward=reward, next_observation=next_state,
                   terminal=terminal, env_info={})

                state = next_state # set state for next iteration
                
             
            self.episodeCumulativeRewards.append(episodeRewards)
            self.eachEpisodeCumuRewards.append(episodeContRewards) # list of a list

            self.logData() # log data
            self.replay.terminate_episode() # run replay end sequence
  
    def plot_rewards(self, filename="reward_plot.png"):
        """ Save cumulative rewards plots to a file. """
        print('plot')
        plt.figure(figsize=(12, 5))

        # Cumulative Rewards per Step
        plt.subplot(1, 2, 1)
        plt.plot(self.cumulativeRewards, label="Cumulative Reward per Step", color='blue')
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward per Step")
        plt.legend()
        plt.grid()

        # Cumulative Rewards per Episode
        plt.subplot(1, 2, 2)
        plt.plot(self.episodeCumulativeRewards, label="Cumulative Reward per Episode", color='red')
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward per Episode")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def initialize_episode(self):
        """ Initializations required before the first episode.

        Should be called before the first episode of a new design is
        executed. Resets variables such as _data_rewards for logging purposes
        etc.

        """
        #self._rl_alg.initialize_episode(init_networks = True, copy_from_gobal = True)
        self.rl_alg.episode_init()
        self.replay.reset_individual_buffer()

        self.data_rewards = []
        self.episode_counter = 0

    
    def first_train_op(self):
        print('in first train op')
        iterations = self.episode_iterations 
        self.data_design_type = 'Optimized'

        # set up rewards file
        
        #self.episodeFilename = "RewardsEachEpisode_Design{}".format(str(self.design_counter))
        #self.episodeFilename = self.episodeFilename+self.date

        self.initialize_episode()
        
        print(f'design counter at {self.design_counter}')
        if self.design_counter == self.num_init_designs: # change this to mathc num init designs #SnakeEnv.get_number_of_init_designs: # if first time after init design loop
         
            self.env.reset()
            
            self.optimized_params = [2.653, 1.280, 2.385, 3.191, 1.485, 2.175, .542] # can set initial parameters yourself
            # or can: self.optimized_params = SnakeEnv.get_random_design()
          

            q_network = self.rl_alg.get_q_network(self.networks['population'])
            policy_network = self.rl_alg.get_policy_network(self.networks['population'])
            self.cost, self.optimized_params = self.do_alg.optimize_design(design=self.optimized_params, q_network=q_network, policy_network=policy_network)
            self.optimized_params = list(self.optimized_params)
            print('OPTIMIZED PARAM NEW DESIGN: ', self.optimized_params)
            print('COST: ', self.cost)
        



    def train_loop(self):
        """ Runs the Fast Evolution through Actor-Critic RL algorithm.

        First the initial design loop is executed in which the rl-algorithm
        is exeuted on the initial designs. Then the design-optimization
        process starts.
        It is possible to have different numbers of iterations for initial
        designs and the design optimization process.
        """
       
        iterations = self.episode_iterations 
        self.data_design_type = 'Optimized'
        self.initialize_episode()
        SnakeEnv.set_new_design(self.optimized_params)

        # Reinforcement Learning
        for episode in range(iterations):
            print('IN TRAINING LOOP')
            self.currEp = episode
            self.train_single_iteration()
        
            #self.plot_rewards()

        # Design Optimization
        print(f'design counter at {self.design_counter}')
        if self.design_counter >= self.num_init_designs:
            self._data_design_type = 'Optimized'
            q_network = self.rl_alg.get_q_network(self.networks['population'])
            policy_network = self.rl_alg.get_policy_network(self.networks['population'])
            self.cost, self.optimized_params = self.do_alg.optimize_design(design=self.optimized_params, q_network=q_network, policy_network=policy_network)
            self.optimized_params = list(self.optimized_params)
            self.design_counter += 1 # another design
            print('NEW DESIGN PARAMETERS: ',self.optimized_params)
            print('COST: ', self.cost)
        #else: # randomize next design
        #    self._data_design_type = 'Random'
        #    self.optimized_params = SnakeEnv.get_random_design()
        #    self.optimized_params = list(self.optimized_params)

        
        self.design_counter += 1 # another design
        
            
    def train_single_iteration(self):
        
        self.replay.set_mode("species")
        self.collect_training_experience() # collect data
        
        if self.design_counter >= 3: # only train population afer certain number of designs, in this case 3
            train_pop = True
        else:
            train_pop = False
        
        print('train single iteration check if counter > episodes before training')
        if self.episode_counter > self.episodes_before_training: # can start training, have filled buffer
            print('counter > episodes')
            q1loss, q2loss, policyloss, popq1loss, popq2loss, poppolicyloss = self.rl_alg.single_train_step(train_ind=True, train_pop=train_pop) # train one step
            
            #log data on lists
            self.q1loss.extend(q1loss)
            self.q2loss.extend(q2loss)
            self.policyloss.extend(policyloss) 
            self.popq1loss.extend(popq1loss)
            self.popq2loss.extend(popq2loss)
            self.poppolicyloss.extend(poppolicyloss)
            self.epListLoss.extend([self.currEp]*len(q1loss))
        self.logTrainLoss() # log data
        self.episode_counter += 1

        print(f'episode counter at: {self.episode_counter}')
        # evaluate policy
        self.evaluate_policy()

        self.save_networks()
      

    def initial_design_loop(self):
        """ The initial training loop for initial designs.

        The initial training loop in which no designs are optimized but only
        initial designs, provided by the environment, are used.

        Args:
            iterations: Integer stating how many training iterations/episodes
                to use per design.

        """
        self.data_design_type = 'Initial'
        params = SnakeEnv.init_design_parameters[self.design_counter] # choose design based on in which design cycle we are

        SnakeEnv.set_new_design(params)
        self.initialize_episode() 

        
        for _ in range(self.episode_counter, self.episode_iterations): # train motor controls for this design iteration #added self.episode_counter
            self.currEp = _
            print('in initial design loop')
            self.train_single_iteration()

            print(f'range {range(self.episode_counter, self.episode_iterations)}')
        self.design_counter+= 1
        
        return
          
    def evaluate_policy(self):
        """ Evaluates the current deterministic policy.

        Evaluates the current policy in the environment by unrolling a single
        episode in the environment.
        The achieved cumulative reward is logged.
        """
        # can add a policy rollout here
        pass
       
    def save_networks(self):
        """ Saves the networks on the disk.
        """
         # TODO: Edit this to store more efficiently
       
        torch.save(self.rl_alg._ind_policy, 'results/ind_policy_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._ind_qf1, 'results/ind_qf1_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._ind_qf2, 'results/ind_qf2_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._ind_qf1_target, 'results/ind_qf1_tar_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._ind_qf2_target, 'results/ind_qf2_tar_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))


        torch.save(self.rl_alg._pop_policy, 'results/pop_policy_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._pop_qf1, 'results/pop_qf1_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._pop_qf2, 'results/pop_qf2_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._pop_qf1_target, 'results/pop_qf1_tar_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._pop_qf2_target, 'results/pop_qf2_tar_{}_Design{}_ep{}.pt'.format(self.date, self.design_counter, self.episode_counter))
     
    
    def logData(self):
        xPositionList, yPositionList = SnakeEnv.returnOptiXList()
        rewardDF = pd.DataFrame()
        rewardDF['Episode'] = self.epList
        rewardDF['Timestep'] = self.timesteps
        rewardDF['X_Position']= xPositionList # added this, need to see if it works
        rewardDF['Y_Position']= yPositionList # added this, need to see if it works
        rewardDF['Rewards'] = self.timestepRewards
        rewardDF['Cumulative_Rewards'] = self.cumulativeRewards

        # log state variablesmotor_and_coadaptation/CoadaptationCode/train_coadapt.py
        rewardDF['Motor1_Action'] = self.actionList[0]
        rewardDF['Motor2_Action'] = self.actionList[1]
        rewardDF['Motor3_Action'] = self.actionList[2]
        rewardDF['Motor4_Action'] = self.actionList[3]
        rewardDF['Motor5_Action'] = self.actionList[4]
        rewardDF['Motor6_Action'] = self.actionList[5]
      

        # log state variables
        rewardDF['X_State'] = self.stateList[0]
        rewardDF['Y_State'] = self.stateList[1]
        rewardDF['Z_State'] = self.stateList[2]
        #rewardDF['X_Heading'] = self.stateList[3]
        rewardDF['Y_Heading'] = self.stateList[3]
        #rewardDF['Z_Heading'] = self.stateList[5]
        rewardDF['Motor1_Pos'] = self.stateList[4]
        rewardDF['Motor2_Pos'] = self.stateList[5]
        rewardDF['Motor3_Pos'] = self.stateList[6]
        rewardDF['Motor4_Pos'] = self.stateList[7]
        rewardDF['Motor5_Pos'] =  self.stateList[8]
        rewardDF['Motor6_Pos'] =  self.stateList[9]
        rewardDF['Plate1'] =  self.stateList[10]
        rewardDF['Plate2'] =  self.stateList[11]
        rewardDF['Plate3'] =  self.stateList[12]
        rewardDF['Plate4'] =  self.stateList[13]
        rewardDF['Plate5'] =  self.stateList[14]
        rewardDF['Plate6'] =  self.stateList[15]
        rewardDF['Plate7'] =  self.stateList[16]

        rewardDF.to_csv(self.filename, index=False)

    def logTrainLoss(self):
        lossDF = pd.DataFrame()
        lossDF['Episode'] = self.epListLoss
        lossDF['Ind_Q1_Loss'] = self.q1loss
        lossDF['Ind_Q2_Loss'] = self.q2loss
        lossDF['Ind_Policy_Loss'] = self.policyloss

         
        lossDF['Pop_Q1_Loss'] = self.popq1loss
        lossDF['Pop_Q2_Loss'] = self.popq2loss
        lossDF['Pop_Policy_Loss'] = self.poppolicyloss
        lossDF.to_csv(self.lossFilename, index=False)

    def load_networks(self, path):
        # can create code here to load networks
        pass


    def logEpisodeRewards(self):
        # FUNCTION NOT USED ANYMORE

        # set up rewards files
        name = "EpisodeRewards_Design{}".format(str(self.design_counter))
        self.episodeFilename = self.date+name

        # saving whole episode's cumulative rewards
        rewardDF = pd.DataFrame(self.episodeCumulativeRewards, columns=['Rewards'])
    
        rewardDF.to_csv(self.episodeFilename)

    def logTimestepRewards(self, episode):
        # FUNCTION NOT USED ANYMORE
        name = "TimestepRewards_Design{}".format(str(self.design_counter))
        self.timestepFilename = self.date+name

        name = "TimestepCumulativeRewards_Design{}".format(str(self.design_counter))
        self.timestepCumulativeFilename = self.date+name

        print('IN PLOT REWARDS')

        rewardDF = pd.DataFrame()
        rewardDF['Episode {} Rewards'.format(str(episode))] = self.timestepRewards
        rewardDF.to_csv(self.timestepFilename)

        rewardDFCumu = pd.DataFrame()
        rewardDFCumu['Episode {} Cumulative Rewards'.format(str(episode))] = self.timestepCumuRewards
        rewardDF.to_csv(self.timestepCumulativeFilename)

        # saving whole episode's cumulative rewards
        rewardDF = pd.DataFrame(self.episodeCumulativeRewards, columns=['Rewards'])
        rewardDF.to_csv(self.filename)
        # saving each episode cumulative reward trends
        episodeRewardDF = pd.DataFrame()
        for i in range(len(self.eachEpisodeCumuRewards)):
            episodeRewardDF[str(i)] = np.array(self.eachEpisodeCumuRewards(i))
        episodeRewardDF.to_csv(self.episodeFilename)

        # saving each episode cumulative reward trends
        episodeRewardDF = pd.DataFrame()
        for i in range(len(self.eachEpisodeCumuRewards)):
            episodeRewardDF[str(i)] = np.array(self.eachEpisodeCumuRewards(i))
        episodeRewardDF.to_csv(self.episodeFilename)

    def passLocks(self, oLock, mLock):
        # pass locks into the environment  
        SnakeEnv.passLocksToEnv(oLock, mLock)
        
    def optiPos(self, stopEvent):
        # to run on thread and interact with snake environment
        while True:   
            SnakeEnv.optiPos()
            if stopEvent.is_set():
                break
        

    def motorPos(self, stopEvent):
        # to run on thread and interact with snake environment
        while True:
            SnakeEnv.motorPos()
            if stopEvent.is_set():
                break
    
    from itertools import tee


if __name__ == '__main__':

    filename = "Design10CoptPopNetFeb20"
    gc.collect()
    gc.set_threshold(0)

    startTrainingSession = False # change this to false if not first time running training cycle
    stopEvent = threading.Event() # event to stop threads

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "rb") as picklefile:
            trainingObj = pickle.load(picklefile)
        picklefile.close()

    else:
        print("Pickle file is missing or empty. Starting first time training")
        stopEvent = threading.Event()

        # need to create new object since first time running code
        trainingObj = Train()
        optiLock = threading.Lock()
        motorLock = threading.Lock()
        trainingObj.passLocks(optiLock, motorLock)


    # run object from current/last state
    motorThread = threading.Thread(target=trainingObj.motorPos, args=(stopEvent,)) 
    optiThread = threading.Thread(target=trainingObj.optiPos, args=(stopEvent,))
    trainingloopThread = threading.Thread(target=trainingObj.run, args=(stopEvent,))


    # start threads
    motorThread.start()
    optiThread.start() 
    trainingloopThread.start() # start this thread last so can populate motor and opti data
    trainingloopThread.join()
    
    # picklefile = open('Design10CoptPopNetFeb20', 'wb')# open pickle file you want to dump/write to 
    # print(type(trainingObj))
    # pickle.dump(trainingObj, picklefile) # save object to file
    # picklefile.close() # close file
