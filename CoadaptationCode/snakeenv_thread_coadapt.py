import gymnasium
from gymnasium import spaces
import numpy as np
import motorssynced
import optitrack
import threading
import math
import random
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.interpolate import interp1d
import os
import copy
from datetime import datetime

gymnasium.envs.register(
    id = "SnakeRobot",
    entry_point = f"{__name__}:SnakeEnv",
    max_episode_steps = 150,  # maybe come back and change
    reward_threshold = 1000,
    
)
global optiPos, motorPos

class SnakeEnv(gymnasium.Env):
    # static variables so can be accessed between static and non static methods
    optiPosition = []
    motorPosition = []
    optiXTrack = []
    optiYTrack = []
    prevPos = [0,0,0]
    optiRelPos = []
    motors = motorssynced.MotorsSynced()
    opti = optitrack.Optitrack()
    motorLock = threading.Lock()
    optiLock = threading.Lock()

    bla = time.time()


    '''
       Robot has 6 motors and 7 snake segments
       Action Space: 6
       Observation Space: 12 from snake robot + 7 from design 
    '''

    # setting up design framework
    current_design = [1.80] * 7 # put initial length in here, 6 links right now!
    design_parameter_bounds = [(.45, 3.60)]*7 # the bounds of how small and large a variable can be 
        
    # init_design_parameters = [
    #         [1.0] * 6
    #         [.5, .5, .5, .5, .5, .5]
    #         [.5, 1, .5, 1, .5, 1]
    #         [.75, .5, .75, .5, 1, 1]
    #     ] # NOTE: Change these depending on the design I am going to use

    init_design_parameters = [
        [1.80] * 7,
        [.60] * 7,
        [2.70] * 7,
        [1.80, .60, 2.70, 1.80, .60, 2.70, 1.80],
        [2.653, 1.280, 2.385, 3.191, 1.485, 2.175, .542],

    ] # NOTE: Change these depending on the design I am going to use

    config_numpy = np.array(current_design)
    
    
    design_dims =  list(range(17-len(current_design), 17)) # 12+7
    print('design dimensions!', design_dims)
    
    def __init__(self):
    
        self.rewardScale = 100 # scale rewards
        self.motorMin = 1422 #1500 #1422
        self.motorMax = 2674 #2500 #2673
       

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10+7,), dtype= 'float32') # data type is float32
        
        self.action_space = spaces.Box(low=self.motorMin, high=self.motorMax, shape=(6,), dtype='float32') # continuous action space
        #self.action_space = spaces.Box(low=self.motorMin, high=self.motorMax, shape=(18,), dtype='float32')

        self.targetPositionX = 100.0 # position that can't be reached, think about changing or getting rid of this

               
        # init other things
        # moved these class declarations to static
        self.motors = motorssynced.MotorsSynced()
        self.opti = optitrack.Optitrack()
        #self.opti.optiTrackInit()

        self.currPosition =[]
        self.newAction = []

        self.reward = 0
        self.prevDist = 0
        self.prevPos = 0 

        self.distList = []
        self.rewardList = []
        self.xPosList = []
        self.i = 0

        # data frame for logging data
        self.df = pd.DataFrame(columns=['Action Sent','Opti Position', 'Motor Position','Reward'])

        # set up files
        self.filename = "Training_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
     
    def step(self, action):

        num_timesteps = 1  
        assert len(action) == 6 * num_timesteps, f"Action space must now be {6 * num_timesteps} values ({6} motors × {num_timesteps} timesteps)."

        # Split action into multiple consecutive timesteps
        actions = [action[i * 6: (i + 1) * 6] for i in range(num_timesteps)]

        for sub_action in actions:
            actionForMotors = self.denormalizeAction(sub_action)
            print(actionForMotors)
            self.writeAction(actionForMotors)

        # Wait 0.5s before retrieving the next batch of actions
        for i in range(3):
            nextObs = self._get_obs()
        #time.sleep(.2)

        for i in range(5):
            nextObs = self._get_obs()
            #print(nextObs)
        tmp_pos = copy.deepcopy(nextObs)
        nextObs[0] = nextObs[0] - self._prev_obs[0]
        nextObs[1] = nextObs[1] - self._prev_obs[1]
        nextObs[2] = nextObs[2] - self._prev_obs[2]

        while nextObs[0] == 0. and nextObs[1] == 0.:
            nextObs = self._get_obs()
            print(nextObs)
            tmp_pos = copy.deepcopy(nextObs)
            nextObs[0] = nextObs[0] - self._prev_obs[0]
            nextObs[1] = nextObs[1] - self._prev_obs[1]
            nextObs[2] = nextObs[2] - self._prev_obs[2]
            print("im here")
            print(nextObs[0], nextObs[1])

        self._prev_obs = tmp_pos

        # Log global positions
        SnakeEnv.optiXTrack.append(SnakeEnv.optiRelPos[0])  # global x position of robot
        SnakeEnv.optiYTrack.append(SnakeEnv.optiRelPos[2])  # global y position of robot

        # extract X position
        currXPos = nextObs[0]  # opti X position of the robot

        # # reward forward movement
        reward = (currXPos - self.prevPos)
        #print("global pos")
        #print(global_pos)

        # max_distance = self.targetPositionX - (-0.7 - 20) 
        # distance = abs(self.targetPositionX - currXPos)
        # reward = np.exp(1 - (distance / max_distance))

        # check if the goal is reached
        terminated = currXPos > self.targetPositionX

        truncated = False
        info = {'info': 0}

        print(f"Reward: {reward}")

        # Log data
        self.df.loc[len(self.df.index)] = [actionForMotors, nextObs[0:6], nextObs[6:-1], reward]

        # update previous position and action
        self.prevPos = currXPos

        # add design info to observation

        nextObs = np.append(nextObs, SnakeEnv.config_numpy)
        print(f"Observation: {nextObs}")

        return np.array(nextObs), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        # returns: observation of the initial state
        print('in reset')
       
        
        super().reset(seed=seed)  # this is needed for cutom environments according to AI Gym

        input('Reset robot then press a button to continue') # used to pause to reset robot position
   


        SnakeEnv.motorLock.acquire()
        SnakeEnv.motors.setMotorSpeed() # set speed here so if motor torques and reset power the speed gets reset
        time.sleep(.5)
        SnakeEnv.motorLock.release()
        # time.sleep(.5)
        print('motor speeds set')


        # choose starting position of robot motors
        #startPos = random.sample(range(self.motorMin, self.motorMax), 6)
        startPos = [2048, 2048, 2048, 2048, 2048, 2048]
        self.writeAction(startPos)
        SnakeEnv.disableMotorTorque()
        # choose new goal position? could randomize target position?
        # self.targetPosition = 
        
        time.sleep(1)   

        # to fill data for reset

        """
        SnakeEnv.optiLock.acquire()
        SnakeEnv.prevPos = SnakeEnv.optiPosition[0:3]
        SnakeEnv.optiLock.release()
        """

        # return current observation
        print("about to observe")  
        observation = self._get_obs(initial=True)
        observation[0] = 0.
        observation[1] = 0.
        observation[2] = 0.
        SnakeEnv.enableMotorTorque()
        # DO NOT UNCOMMENT IN

        print('Observation: ', observation)
        self.starting_position = observation[0]
        self.prevPos = observation[0] # x position of observation

        self._prev_obs = copy.deepcopy(observation)


        info = {'info': 0}

        observationfull = np.append(observation, SnakeEnv.config_numpy)

        print('full observation', observationfull)
        return (np.array(observationfull), info)

    def render(self):
        # graphical window
        # leave empty if not giving user a way to visualize 
        pass

    def close(self):
        # use this to close any files or at the end of sequence
        pass

    def seed(self, seed = None):
        # can use this method to create a random seed
        pass

    def _get_obs(self, initial=False):
        # read agent x,y,z,etc and target goal pos
        # return {"agent": self._agent_location, "target": self._target_location}
        
        self.agentPos = self.getPosition(initial)
       
        return [*self.agentPos]
    
    def getPosition(self, initial):
       
        # motorPos = self.motors.readPos()
        # optiPos = self.opti.optiTrackGetPos() # currently returning x, y, z
        # self.currPosition = [*optiPos, *motorPos]  
        
 
        SnakeEnv.motorLock.acquire()
        SnakeEnv.optiLock.acquire()

        if initial == True:
            SnakeEnv.prevPos = SnakeEnv.optiPosition[0:3]

        #optiPositionCoord = [(curr- prev)*100 for curr, prev in zip(SnakeEnv.optiPosition[0:3], SnakeEnv.prevPos)] # adjusting position to measure previous
        optiPositionCoord_global = [(curr)*100 for curr, _ in zip(SnakeEnv.optiPosition[0:3], SnakeEnv.prevPos)]
        #optiAngle = [i/100 for i in SnakeEnv.optiPosition[3:6]] # only accessing y heading 
        optiAngle = SnakeEnv.optiPosition[4]/100
        # print('MOTOR POS', SnakeEnv.motorPosition)

        #while SnakeEnv.motorPosition == []:
        #    SnakeEnv.motorLock.release()
        #    time.sleep(.001)
        #    SnakeEnv.motorLock.acquire()
        #print('CHANGE')
        self.currPosition = [*optiPositionCoord_global, optiAngle, *SnakeEnv.motorPosition] # reads static variables that are being updated in the threads 
        
        while SnakeEnv.motorPosition == []:
            SnakeEnv.motorLock.release()
            time.sleep(.001)
            SnakeEnv.motorLock.acquire()
            self.currPosition = [*optiPositionCoord_global, optiAngle, *SnakeEnv.motorPosition]

        SnakeEnv.prevPos = SnakeEnv.optiPosition[0:3]

        SnakeEnv.motorLock.release()
        SnakeEnv.optiLock.release()
        
        time.sleep(.001)
        return self.currPosition
    
    def writeAction(self, actionToWrite):
        posTo = actionToWrite
        print('POSITION TO', posTo)
        SnakeEnv.motorLock.acquire()
        SnakeEnv.motors.writePos(posTo)
        SnakeEnv.motorLock.release()

        time.sleep(.3) # sleep to allow motors to get to position


    def getTorque(self):
        motorTor = self.motors.readTorque(self.motorLock)
        return motorTor
    
    def denormalizeAction(self, action):
        
        motorMax = self.motorMax
        motorMin = self.motorMin

        mapping = interp1d([-1, 1], [motorMin, motorMax])
        mappedList = [int(mapping(i)) for i in action]

        return mappedList

    '''
        The following methods are static so they can be accessed from outside environment wrapper to edit parameters with threading 
    '''
    @staticmethod
    def passLocksToEnv(oLock, mLock):
        # function to pass locks into this environment
        SnakeEnv.optiLock = oLock
        SnakeEnv.motorLock = mLock
        return
    
    @staticmethod
    def optiPos():
        SnakeEnv.optiLock.acquire()
        SnakeEnv.optiRelPos, heading = SnakeEnv.opti.optiTrackGetPos()
        #print(time.time() - SnakeEnv.bla)
        SnakeEnv.optiPosition = [*SnakeEnv.optiRelPos, *heading]
        #print(SnakeEnv.optiPosition)
        SnakeEnv.optiLock.release()
        time.sleep(.001) # changed from .008
        #SnakeEnv.bla = time.time()
        return
    
    @staticmethod
    def motorPos():
        SnakeEnv.motorLock.acquire()
        SnakeEnv.motorPosition = SnakeEnv.motors.readPos()
        # print('In thread', SnakeEnv.motorPosition)
        SnakeEnv.motorLock.release()
        time.sleep(.001)
        return
    
    @staticmethod
    def returnOptiXList():
        return SnakeEnv.optiXTrack, SnakeEnv.optiYTrack
    

    @staticmethod
    def disableMotorTorque():
        SnakeEnv.motorLock.acquire()
        SnakeEnv.motorPosition = SnakeEnv.motors.disableTorque()
        SnakeEnv.motorLock.release()
        #time.sleep(.005)
        return   

    @staticmethod
    def enableMotorTorque():
        SnakeEnv.motorLock.acquire()
        SnakeEnv.motorPosition = SnakeEnv.motors.enableTorque()
        SnakeEnv.motorLock.release()
        #time.sleep(.005)
        return   

    @staticmethod
    def set_new_design(design):
        SnakeEnv.current_design = design
        SnakeEnv.config_numpy = np.array(design)
      
    @staticmethod 
    def get_random_design():
        optimized_params = np.random.uniform(low=SnakeEnv.design_parameter_bounds[0][0], high=SnakeEnv.design_parameter_bounds[0][1], size=7)
        return optimized_params
      
    @staticmethod
    def get_current_design():
        return copy.copy(SnakeEnv.current_design)
  
    @staticmethod
    def get_design_dimensions():
        return copy.copy(SnakeEnv.design_dims)

    @staticmethod
    def get_number_of_init_designs():
        print('NUM DESIGNS', len(SnakeEnv.init_design_parameters))
        return len(SnakeEnv.init_design_parameters)
        

    