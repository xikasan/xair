# coding: utf-8

import gym
import numpy as np
import xtools as xt
import xsim
from .base import BaseEnv
from ..models.lvaircraft import LVAircraft


class LVAircraftPitchV0(BaseEnv):

    def __init__(
            self,
            dt,
            range_target=xt.d2r([-10, 10]),
            target_period=10.0,
            dtype=np.float32,
            name="LVAircraftPitchV0"
    ):
        super().__init__(dt, dtype=dtype, name=name)

        # dynamics model
        self._model = LVAircraft(
            dt,
            range_throttle=[-10, 10],
            range_elevator=xt.d2r([-5, 5]),
            name=name+"/model"
        )
        self.IX_T = 0
        self.IX_q = 1
        self.IX_C = 2

        # command generator
        self.range_target = range_target
        target_width = np.max(range_target) - np.max(range_target) / 2
        target_center = np.sum(range_target) / 2
        self._ref = xsim.RectangularCommand(
            period=target_period,
            amplitude=target_width,
            bias=target_center)

        # env spaces
        self.action_space = self.generate_space(self._model.act_low, self._model.act_high)
        self._obs_low = np.concatenate([
            self._model.obs_low[[self._model.IX_T, self._model.IX_q]],
            [np.min(self.range_target)]
        ])
        self._obs_high = np.concatenate([
            self._model.obs_high[[self._model.IX_T, self._model.IX_q]],
            [np.max(self.range_target)]
        ])
        self.observation_space = self.generate_space(self._obs_low, self._obs_high)

        # env variables
        self.current_time = 0.
        self.viewer = None

    def reset(self):
        self.current_time = 0.
        self._model.reset()
        return self.get_observation()

    def step(self, action):
        action = np.asanyarray(action).astype(self.dtype)
        assert action.shape == (2,), \
            "action size is expected 2 and only single action is accepted."
        self._model(action)
        self.current_time = xt.round(self.current_time + self.dt, 2)

        return self.get_observation(), self.calc_reward(), False, {}

    def calc_reward(self, ):
        observation = self.get_observation()
        reward_T = observation[self.IX_T] - observation[self.IX_C]
        reward_T = reward_T / self._ref.amplitude
        reward_q = observation[self.IX_q]
        reward = - reward_T ** 2 - reward_q ** 2
        return reward

    def get_observation(self):
        return np.array([
            self._model.get_theta(),
            self._model.get_q(),
            self._ref(self.current_time)
        ]).astype(self.dtype)

    def get_target(self):
        return self.get_observation()[self.IX_C]

    def get_state(self):
        return self.get_observation()[[self.IX_T, self.IX_q]]

    def get_time(self):
        return self.current_time


class LVAircraftPitchV1(LVAircraftPitchV0):

    def __init__(
            self,
            dt,
            tau=1.0,
            range_target=xt.d2r([-10, 10]),
            target_period=10.0,
            dtype=np.float32,
            name="LVAircraftPitchV0"
    ):
        super().__init__(
            dt,
            range_target=range_target,
            target_period=target_period,
            dtype=dtype,
            name=name
        )

        self._flc = xsim.Filter2nd(dt, tau)

    def step(self, action):
        super().step(action)
        self._flc(self._ref(self.current_time))

        return self.get_observation(), self.calc_reward(), False, {}

    def reset(self):
        self.current_time = 0.0
        self._model.reset()
        self._flc.reset(self._ref(self.current_time))
        return self.get_observation()

    def get_observation(self):
        return np.array([
            self._model.get_theta(),
            self._model.get_q(),
            self._flc.get_state()[0]
        ], dtype=self.dtype)

    def get_target(self):
        return self._flc.get_full_state().astype(dtype=self.dtype)
