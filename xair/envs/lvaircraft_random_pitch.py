# coding: utf-8

import gym
import xsim
import numpy as np
import xtools as xt
from .base import BaseEnv
from ..models.lvaircraft import LVAircraft
from .lvaircraft_pitch import LVAircraftPitchV3


class LVAircraftPitchV4(LVAircraftPitchV3):

    IX_T = 0
    IX_q = 1
    IX_r = 2
    IX_dt = 0
    IX_de = 1

    def __init__(
            self,
            dt=1/100,
            target_range=xt.d2r([-10, 10]),
            target_period=10.0,
            fail_mode="nomal",
            fail_range=[0.2, 0.7],
            dtype=np.float32,
            name="LVAircraftRandomPitchV0"
    ):
        super().__init__(
            dt,
            target_range=target_range,
            target_period=target_period,
            fail_mode=fail_mode,
            fail_range=fail_range,
            dtype=dtype,
            name=name
        )

        target_width = (np.max(target_range) - np.min(target_range)) / 2
        self._ref = xsim.PoissonRectangularCommand(
            max_amplitude=target_width,
            interval=target_period
        )
