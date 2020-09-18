# coding: utf-8

import gym
import numpy as np
from cached_property import cached_property


class BaseEnv(gym.Env):

    def __init__(self, dt, dtype=np.float32, name="BaseEnv"):
        super().__init__()

        self.dt = dt
        self.name = name
        self.dtype = dtype
        self._act_low = 0
        self._act_high = 0

    def step(self, action):
        raise NotImplementedError()

    def resset(self):
        raise NotImplementedError()

    @cached_property
    def action_size(self):
        return self.action_space.high.size

    @cached_property
    def observation_size(self):
        return self.observation_space.high.size

    def generate_space(self, low, high):
        high = np.array(high, dtype=self.dtype) if type(high) is list else self.dtype(high)
        low  = np.array(low,  dtype=self.dtype) if type(low)  is list else self.dtype(low)
        return gym.spaces.Box(high=high, low=low)

    def clip_action(self, action):
        return np.clip(action, self._act_low, self._act_high)
