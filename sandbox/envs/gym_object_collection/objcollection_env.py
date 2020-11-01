import sys
from random import randint, randrange
import numpy as np
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class ObjCollectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ObjCollectionEnv, self).__init__()

    def init_collection(self, bias=False, num_props=None, num_prop_types=None, num_selections=5, horizon=20):
        assert(num_props is not None)
        assert(num_prop_types is not None)

        self.bias = bias

        self.steps = 0
        self.horizon = horizon

        self.num_props = num_props
        self.num_prop_types = num_prop_types

        self.num_selections = num_selections
        self.num_actions = num_selections + 1

        #TODO Figure out how to generate this
        # Bias towards second row and 3 column
        #self.rewards = np.array([[-1.0, -1.0, 0.5], [1.0, 1.0 ,5.0], [-1.0, -1.0, 0.5], [-1.0, -1.0, 0.5]])
        self.rewards = np.ones((num_props, num_prop_types)) * -1.0
        self.best_prop = np.random.randint(num_props, size=1)[0]
        self.best_prop_type = np.random.randint(num_prop_types, size=1)[0]
        #self.rewards[:,self.best_pro_type] = 0.5
        self.rewards[self.best_prop] = 1.0
        self.rewards[self.best_prop][self.best_prop_type]  = 1.5

        def get_bias_vec():
            val = list(range(1, 20))
            dist = np.random.choice(val,self.num_prop_types)
            dist = dist / np.sum(dist)
            return dist

        self.bias_prop_vec = np.array([get_bias_vec() for _ in range(self.num_props)])

        print(f'Best property: {self.best_prop}')
        print(f'Best property type: {self.best_prop_type}')

        self._action_space = spaces.Discrete(self.num_actions)
        self._obs_space = None
        self._just_reset = False

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._obs_space

    def _random_objects(self):
        _object = np.zeros((self.num_props, self.num_prop_types))

        if not self.bias:
            random.choice(_object)[randrange(len(_object[0]))] += 1
        else:
            prop = random.choice(list(range(self.num_props)))
            indices = list(range(self.num_prop_types))
            type = np.random.choice(indices, 1, p=self.bias_prop_vec[prop])
            _object[prop][type] += 1
        return _object

    def _reward(self, action):
        """
        Compute single-timestep reward after having taken the action specified
        by `action`.
        """
        if action == (self.num_actions - 1):
            # Last Action, generates all new objects to select
            # reward is zero
            reward = -0.5
        else:
            _object = self._obs_space[action]
            reward = np.sum(np.multiply(_object, self.rewards))
        return reward

    def _take_action(self, action, shuffle=True):
        if action == (self.num_actions - 1):
            # Last Action, generates all new objects to select
            bag_state = self._obs_space[-1]
            selection_state = [self._random_objects() for _ in range(self.num_selections)]
            self._obs_space = np.stack(selection_state + [bag_state])
        else:
            # Add selected object to bag
            _object = self._obs_space[action]
            self._obs_space[-1] += _object

            # Remove selected object and add new one
            self._obs_space[action] = self._random_objects()

            if shuffle:
                obs_indices = np.array(range(self._obs_space.shape[0]-1))
                np.random.shuffle(obs_indices)
                obs_indices = np.append(obs_indices, self._obs_space.shape[0]-1)
                self._obs_space = self._obs_space[obs_indices]
        return self._obs_space

    def step(self, action):
        self.steps+=1

        reward = self._reward(action)
        obs = self._take_action(action)
        done = True if self.steps > self.horizon else False
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.steps = 0

        bag_state = np.zeros((self.num_props, self.num_prop_types)).tolist()
        selection_state = [self._random_objects() for _ in range(self.num_selections)]
        self._obs_space = np.stack(selection_state + [bag_state])

        obs = self._obs_space
        return obs


    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        obs = self._obs_space
        return obs

    def seed(self, seed):
        random.seed(seed)
        np.random.seed
