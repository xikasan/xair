# coding: utf-8

import xsim
import xtools as xt
import numpy as np
import enum


class FailMode(enum.Enum):

    NORMAL = enum.auto()
    STABILITY_LOSS = enum.auto()
    GAIN_REDUCTION = enum.auto()

    def __init__(self, value):
        self._val = value - 1
        self._mag = 1

    def set_val(self, val):
        self._mag = val
        return self

    def get_val(self):
        return self._mag

    @classmethod
    def get_mode(cls, mode):
        mode = mode.upper()
        for key, fail in cls.__members__.items():
            if key == mode:
                return fail
        raise ValueError("Not defined mode name {} is given.".format(mode))


class LVAircraftEx(xsim.BaseModel):

    H0 = 12192  # m
    U0 = 625    # m/s

    # indices
    IX_U = 0
    IX_H = 1
    IX_u = 2
    IX_w = 3
    IX_T = 4
    IX_q = 5
    IX_dt = 0
    IX_de = 1

    _IX_Mw  = [5, 3]
    _IX_Xde = [3, 1]
    _IX_Mde = [5, 1]

    def __init__(
            self,
            dt,
            range_elevator=xt.d2r([-10, 10]),
            range_throttle=[-1, 1],
            dtype=np.float32,
            name="LVAircraft"
    ):
        super().__init__(dt, dtype=dtype, name=name)

        # parameters
        self._A, self._B = self._construct_matrices()

        # state
        self._x = np.zeros(6, dtype=self.dtype)
        # fail
        self._fail_mode = FailMode.NORMAL

        # env params
        # action space
        # U = [throttle, elevator]
        self.act_low  = np.array([np.min(range_throttle), np.min(range_elevator)]).astype(dtype)
        self.act_high = np.array([np.max(range_throttle), np.max(range_elevator)]).astype(dtype)
        # observation space
        self.obs_low, self.obs_high = self.generate_inf_range(6)

    def __call__(self, action):
        action = np.clip(action, self.act_low, self.act_high).astype(self.dtype)
        fn = lambda x: x.dot(self._A) + action.dot(self._B)
        dx = xsim.no_time_rungekutta(fn, self.dt, self._x)
        self._x += dx * self.dt
        return self.get_observation()

    def reset(self):
        self._A, self._B = self._construct_matrices()
        self._x = np.zeros(6, dtype=self.dtype)
        self._x[self.IX_U] = self.U0
        self._x[self.IX_H] = self.H0
        self._fail_mode = FailMode.NORMAL
        return self.get_observation()

    def get_observation(self):
        return self._x.astype(self.dtype)

    def get_state(self):
        return self._x[self.IX_u:].astype(self.dtype)

    def get_H(self):
        return self._x[self.IX_H]

    def get_U(self):
        return self._x[self.IX_U]

    def get_u(self):
        return self._x[self.IX_u]

    def get_w(self):
        return self._x[self.IX_w]

    def get_T(self):
        return self._x[self.IX_T]

    def get_q(self):
        return self._x[self.IX_q]

    def _construct_matrices(self):
        A = np.array([
            #U  H   u        w         Theta     q
            [0, 0,  1,       0,        0,        0],       # H
            [0, 0,  0,       1,        0,        0],       # U
            [0, 0, -0.0225,  0.0022, -32.3819,   0],       # u
            [0, 0, -0.2282, -0.4038,   0,      869],       # w
            [0, 0,  0,       0,        0,        1],       # Theta
            [0, 0, -0.0001, -0.0018,   0,       -0.5518]   # q
        ], dtype=self.dtype)
        B = np.array([
            #dt      de
            [0,      0],        # H
            [0,      0],        # U
            [0.500,  0],        # u
            [0,     -0.0219],   # w
            [0,      0],        # Theta
            [0,     -1.2394]    # q
        ], dtype=self.dtype)
        return A.T, B.T

    def set_fail(self, mode, val=None):
        if val is not None:
            mode.set_val(val)
        self._fail_mode = mode
        if self._fail_mode == FailMode.NORMAL:
            return self
        if self._fail_mode == FailMode.GAIN_REDUCTION:
            self._B[self._IX_Xde[1], self._IX_Xde[0]] *= self._fail_mode.get_val()
            self._B[self._IX_Mde[1], self._IX_Mde[0]] *= self._fail_mode.get_val()
            return self
        if self._fail_mode == FailMode.STABILITY_LOSS:
            self._A[self._IX_Mw[1], self._IX_Mw[0]] *= self._fail_mode.get_val()
            return self

    def get_fail(self):
        return self._fail_mode


class LVAircraft(xsim.BaseModel):

    # indices
    IX_u = 0
    IX_w = 1
    IX_T = 2
    IX_q = 3
    IX_dt = 0
    IX_de = 1

    _IX_Mw  = [3, 1]
    _IX_Xde = [1, 1]
    _IX_Mde = [3, 1]

    def __init__(
            self,
            dt,
            range_elevator=xt.d2r([-10, 10]),
            range_throttle=[-100, 100],
            dtype=np.float32,
            name="LVAircraft",
    ):
        super().__init__(dt, dtype=dtype, name=name)

        # parameters
        self._A, self._B = self.construct_matrices()

        # state
        self._x = np.zeros(4, dtype=self.dtype)
        # fail
        self._fail_mode = FailMode.NORMAL

        # env params
        # action space
        # U = [throttle, elevator]
        self.act_low  = np.array([np.min(range_throttle), np.min(range_elevator)])
        self.act_high = np.array([np.max(range_throttle), np.max(range_elevator)])
        # observation space
        self.obs_low, self.obs_high = self.generate_inf_range(4)

    def __call__(self, action):
        action = np.clip(action, self.act_low, self.act_high).astype(self.dtype)
        fn = lambda x: x.dot(self._A) + action.dot(self._B)
        dx = xsim.no_time_rungekutta(fn, self.dt, self._x)
        self._x += dx * self.dt
        return self._x

    def reset(self):
        self._A, self._B = self.construct_matrices()
        self._x = np.zeros(4, dtype=self.dtype)
        return self.get_state()

    def get_state(self):
        return self._x.astype(self.dtype)

    def get_u(self):
        return self._x[self.IX_u]

    def get_w(self):
        return self._x[self.IX_w]

    def get_theta(self):
        return self._x[self.IX_T]

    def get_q(self):
        return self._x[self.IX_q]

    def construct_matrices(self):
        A = np.array([
            # u        w         Theta     q
            [-0.0225,  0.0022, -32.3819,   0],      # u
            [-0.2282, -0.4038,   0,      869],      # w
            [ 0,       0,        0,        1],      # Theta
            [-0.0001, -0.0018,   0,       -0.5518]  # q
        ], dtype=self.dtype)
        B = np.array([
            #dt      de
            [0.500,  0],        # u
            [0,     -0.0219],   # w
            [0,      0],        # Theta
            [0,     -1.2394]    # q
        ], dtype=self.dtype)
        return A.T, B.T

    def set_fail(self, mode, val=None):
        if val is not None:
            mode.set_val(val)
        self._fail_mode = mode
        if self._fail_mode == FailMode.NORMAL:
            return self
        if self._fail_mode == FailMode.GAIN_REDUCTION:
            self._B[self._IX_Xde[1], self._IX_Xde[0]] *= self._fail_mode.get_val()
            self._B[self._IX_Mde[1], self._IX_Mde[0]] *= self._fail_mode.get_val()
            return self
        if self._fail_mode == FailMode.STABILITY_LOSS:
            self._A[self._IX_Mw[1], self._IX_Mw[0]] *= self._fail_mode.get_val()
            return self

    def get_fail(self):
        return self._fail_mode
