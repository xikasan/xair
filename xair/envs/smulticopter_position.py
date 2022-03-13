# coding: utf-8

import numpy as np
import xtools as xt
import xsim
from typing import List, Union

from .base import BaseEnv
from ..models.multicopter import SimpleMulticopter


class SMulticopterPosition2DV0(BaseEnv):

    position_residual = 5e-2

    def __init__(
            self,
            dt: float = 0.01,
            target_position: List[float] = [1., 2.],
            dtype: np.dtype = np.float32,
            name: str = "SMulticopterPosition2DV0"
    ):
        super().__init__(dt, dtype=dtype, name=name)

        # dynamics model
        self._model = SimpleMulticopter(
            dt,
            dtype=dtype,
            name=name+"/model"
        )
        self.E = self._model.E

        self.target_position = np.asarray(target_position)

        # env spaces
        self.action_space = self.generate_space(self._model.act_low, self._model.act_high)
        self.observation_space = self.generate_space(self._model.obs_low, self._model.obs_high)

        # env variables
        self.viewer = None

    def reset(self):
        self._model.reset()
        return self.get_observation()

    def step(self, action: Union[np.ndarray, List]):
        action = np.asanyarray(action).astype(self.dtype)
        assert action.shape == (4,), \
            "action size is expected 4 and only single action is accepted."
        self._model(action)
        return self.get_observation(), self._calc_reward(), self._is_done, {}

    def _get_reward(self):
        obs = self._model.get_state()
        ix = self._model.IX
        reward_pos = (obs[ix.POS][:2] - self.target_position) / self.target_position
        reward_pos = np.linalg.norm(reward_pos)
        return reward_pos

    def _is_done(self):
        obs = self._model.get_state()
        ix = self._model.IX
        error = np.linalg.norm(obs[ix.POS][:2] - self.target_position)
        return error <= self.position_residual

    def get_observation(self):
        return self._model.get_state().astype(self.dtype)

    def get_target(self):
        return self.target_position
