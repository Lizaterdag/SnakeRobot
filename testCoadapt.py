from rlkit.torch.sac.policies import TanhGaussianPolicy
# from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
import numpy as np
from rl_algorithm import RLAlgorithm
# from rlkit.torch.sac.sac import SACTrainer
from SACTrainer import SACTrainer
import rlkit.torch.pytorch_util as ptu
import torch
import utils
import torch.optim as optim
import threading
from optitrack import Optitrack
from motorssynced import MotorsSynced
import torch
from scipy.interpolate import interp1d
import time
import numpy as np
import pandas as pd
from datetime import datetime

# define locks
motorLock = threading.Lock()
optiPosLock = threading.Lock()
optiPosLock.acquire()
optiPosLock.release()
currentPosLock = threading.Lock()

# define objects
optiObj = Optitrack()
#optiObj.optiTrackInit() # initialize optitrack
motorsObj = MotorsSynced()

# define stop event
stopEvent = threading.Event()

# global variables
optiPos=[]
motorPos=[]
globalXPos = []
globalZPos = []
filename = 'File'
prevPos = [0,0,0]

firstTime = True
torch.use_deterministic_algorithms(True)



# upload policy
#policy = PolicyNetwork(state_dim=17,action_dim=6)
policy = torch.load('/home/liza/SnakeRobotInternal/Design_Networks/ind_policy_2024_06_11-17_18_08_Design5_ep35.pt', map_location=torch.device('cpu')) # may need to change this
print('WHAT IS POLICY', isinstance(policy, torch.nn.Module))
policy.eval()
filename = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'Design1Policy41Trained_Rewards'

def getOptiData():
   while not stopEvent.is_set(): # loop until stop even is set
    global optiPos
    optiPosLock.acquire()

    pos, heading = optiObj.optiTrackGetPos()

    optiPos= [*pos, *heading]

    optiPosLock.release()
    time.sleep(.005)

def getMotorData():
    while not stopEvent.is_set(): # loop until stop event is set
        global motorPos
        motorLock.acquire()
        motorPos = motorsObj.readPos()
        motorLock.release()
        time.sleep(.03)

def runPolicyAndWrite():

    prevPos = 4 # select arbitrary first value
    timestep = 0
    timestepRewards = []
    cumulativeRewards = []
    timesteps  = []


    input('everything set up begin testing press enter')
    #print('test')
    while timestep < 100: # how many steps want policy to execute
        #print('test')
        currState = currentState() # get current data
        #print('test1')
        print('CURRENT STATE', currState)

        # find reward
        reward = currState[0]/100*100# if positive is moving forward, if negative moving backward (not rewarded for this)=
        #reward = np.clip(reward, a_min=0, a_max=50) # clip reward so if move backwards in negative

        # log rewards
        if timestep != 0:
            print('timestep:', timestep)
            print('cumu reward:' , cumulativeRewards[timestep-1]+reward)
            cumulativeRewards.append(cumulativeRewards[timestep-1]+reward)
        else:
            reward = 0
            cumulativeRewards.append(0) # since is first time running
        timestepRewards.append(reward)
        timesteps.append(timestep)
        logRewards(timestepRewards, cumulativeRewards,timesteps)

        global policy
        print(currState)
        currState = torch.tensor(currState)
        currState = currState.to(torch.float32)

        currState = currState[None,:]
  
        #torch.use_deterministic_algorithms(True)
        action, *_ = policy.forward(currState, deterministic=False)
        #action, other= policy(currState, return_logdeterministic=False) 
        print('ACTION', action)
        #print('MEAN',logprob)
        print('OTHER' , *_)
        
    

        
        print(action)
        action = action[0].detach().cpu().numpy()
        print(action)
        motorAction = denormalizeAction(action)
        # print('ACTION', motorAction)
        writeMotors(motorAction) # write new action to motors

        timestep +=1 

    motorLock.acquire()
    motorsObj.disableTorque()
    motorLock.release()
        #time.sleep(.5)
    stopEvent.set() # set stop event
    
def currentState():
    targetPos = 2.5
    # print('motorlock')
    motorLock.acquire()
    # print('poslock')
    optiPosLock.acquire()
    # print('after lock')

    global optiPos, motorPos, prevPos, firstTime
    print('if first time')
    if firstTime:
        firstTime= False
        prevPos = optiPos[0:3]
    
   

    print('optipos',optiPos)
    optiPositionCoord = [(curr- prev)*100 for curr, prev in zip(optiPos[0:3], prevPos)] 
    optiAngle = optiPos[4]/100
    print('optipos pos coord', optiPositionCoord)
    state = [*optiPositionCoord, optiAngle, *motorPos, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
    
    
    prevPos = optiPos[0:3]
    globalXPos.append(prevPos[0])
    globalZPos.append(prevPos[2])
    optiPosLock.release()
    motorLock.release()
    
    time.sleep(.001)

    print('STATE',state)
    return state

def writeMotors(action):
    motorLock.acquire()
    motorsObj.writePos(action)
    motorLock.release()
    time.sleep(0.4)
 
def denormalizeAction(action): 
    # to turn from range -1, 1 to motor value range
    
    motorMax = 2674
    motorMin = 1422

    mapping = interp1d([-1, 1], [motorMin, motorMax])
    mappedList = [int(mapping(i)) for i in action]

    return mappedList
    
def logRewards(rewardList, cumuRewardList, timesteps):
    
    rewardDF = pd.DataFrame()
    rewardDF['Timestep'] = timesteps
    rewardDF['Rewards'] = rewardList
    rewardDF['Cumulative Rewards'] = cumuRewardList
    rewardDF['X Position Global'] = globalXPos
    rewardDF['Z Position Global'] = globalZPos
    rewardDF.to_csv(filename)


if __name__ == '__main__':
    # motorsObj.writePos([1422, 1422, 1422, 1422, 1422, 1422])
    # time.sleep(5)
    motorsObj.writePos([2100, 1900, 2100, 1900, 2048, 1900])
    time.sleep(5)
 
    motorThread = threading.Thread(target=getMotorData)

    optiThread = threading.Thread(target=getOptiData)

    runPolicyThread = threading.Thread(target=runPolicyAndWrite)


    motorThread.start()
    optiThread.start()
    runPolicyThread.start()

    runPolicyThread.join() # wait for this loop to stop executing

