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
            target_range=xt.d2r([-10, 10]),
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
        self.IX_dt = 0
        self.IX_de = 1

        # command generator
        self.target_range = target_range
        target_width = (np.max(target_range) - np.min(target_range)) / 2
        target_center = np.sum(target_range) / 2
        self._ref = xsim.RectangularCommand(
            period=target_period,
            amplitude=target_width,
            bias=target_center)

        # env spaces
        self.action_space = self.generate_space(self._model.act_low, self._model.act_high)
        self._obs_low = np.concatenate([
            self._model.obs_low[[self._model.IX_T, self._model.IX_q]],
            [np.min(self.target_range)]
        ])
        self._obs_high = np.concatenate([
            self._model.obs_high[[self._model.IX_T, self._model.IX_q]],
            [np.max(self.target_range)]
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

    @property
    def target(self):
        return self.get_target()

    @property
    def reference(self):
        return self.get_target()

    def get_state(self):
        return self.get_observation()[[self.IX_T, self.IX_q]]

    @property
    def state(self):
        return self.get_state()

    def get_time(self):
        return self.current_time

    @property
    def time(self):
        return self.get_time()


class LVAircraftPitchV1(LVAircraftPitchV0):

    def __init__(
            self,
            dt,
            tau=1.0,
            target_range=xt.d2r([-10, 10]),
            target_period=10.0,
            dtype=np.float32,
            name="LVAircraftPitchV1"
    ):
        super().__init__(
            dt,
            target_range=target_range,
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
        self._flc.reset()
        return self.get_observation()

    def get_observation(self):
        return np.array([
            self._model.get_theta(),
            self._model.get_q(),
            self._flc.get_state()[0]
        ], dtype=self.dtype)

    def get_target(self):
        return self._flc.get_full_state().astype(dtype=self.dtype)


class LVAircraftPitchV2(LVAircraftPitchV1):

    def __init__(self, *args, name="LVAircraftPitchV2", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def set_fail_mode(self, mode, val=None):
        self._model.set_fail(mode, val)

    def get_fail_mode(self):
        return self._model.get_fail()


class LVAircraftPitchV3(LVAircraftPitchV2):

    def __init__(
            self,
            *args,
            fail_mode="normal",
            fail_range=[0.2, 0.7],
            name="LVAircraftPitchV3",
            **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)
        self.fail_mode = fail_mode
        self.fail_range = fail_range

    def set_fail(self):
        fail_width = np.max(self.fail_range) - np.min(self.fail_range)
        val = np.random.rand() * fail_width + np.min(self.fail_range)
        self.set_fail_mode(self.fail_mode, val=val)
