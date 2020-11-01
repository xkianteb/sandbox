import gym
from gym.spaces import Box
from gym import spaces
import numpy as np
from BPROSFeatures import BPROS
from gym import Wrapper
import ctypes
from ast import literal_eval

class WrapperBPROST(Wrapper):
    def __init__(self, env, mode='bpros'):
        super(WrapperBPROST, self).__init__(env)
        numRows      = 14
        numCols      = 16
        numColors    = 128
        screenHeight = 210
        screenWidth  = 160
        self.bpros = BPROS(screenHeight, screenWidth, numRows, numCols, numColors)
        #self.bprost = BPROST(screenHeight, screenWidth, numRows, numCols, numColors)

    def BPROSFeatures(self):
        pyScreen = self.env.ale.getScreen()
        screen = pyScreen.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.bpros.getActiveFeatures(screen)

        features = np.zeros(21598848)
        features[literal_eval(repr(self.bpros))] = 1
        return  features

    def BPROSTFeatures(self):
        pyScreen = self.env.ale.getScreen()
        screen = pyScreen.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.bprost.getActiveFeatures(screen)

        features = np.zeros(115702400)
        features[literal_eval(repr(self.bprost))] = 1
        return  features

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.BPROSFeatures(), reward, done, info

    def reset(self):
        state = self.env.reset()
        return self.BPROSFeatures()

if __name__ == "__main__":
    env = WrapperBPROST(gym.make('AirRaid-v0'))
    env.reset()
    ep_reward=0
    while True:
        a = env.action_space.sample()
        obs, reward, done, _ = env.step(a)
        print(f'a: {a} | reward: {reward} | obs: {obs.shape}')
        ep_reward += reward
        if done:
            break
    print(f'ep_reward: {ep_reward}')
