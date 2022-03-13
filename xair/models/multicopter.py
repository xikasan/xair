# coding: utf-8

import xsim
import xtools as xt
import numpy as np


class MulticopterIX:
    o1, o2, o3, o4 = 0, 1, 2, 3
    X, Y, Z = 4, 5, 6
    U, V, W = 7, 8, 9
    P, Q = 10, 11
    p, q = 12, 13
    sentinel = 14
    OMG = [0, 1, 2, 3]
    POS = [4, 5, 6]
    VEL = [7, 8, 9]
    ANG = [10, 11]
    ASP = [12, 13]


class SimpleMulticopter(xsim.BaseModel):

    IX = MulticopterIX()

    NUM_ROTOR = 4
    tau = 0.15
    Ixx = 0.060
    Iyy = 0.060
    Izz = 0.12
    M = 2.1
    L = 0.3
    G = 9.81
    omega0 = 275.22349358

    kT = 2.34e-5
    kQ = 7e-7  # 0.03125

    Ew = -1. * kT / M
    Ep = L / np.sqrt(2) * kT / Ixx
    Eq = L / np.sqrt(2) * kT / Iyy
    Er = kQ / Izz
    E = np.array([
        #o1 o2 o3 o4
        [Ew, Ew, Ew, Ew],  # w
        [-Ep, -Ep, Ep, Ep],  # p
        [Eq, -Eq, -Eq, Eq],  # q
        [Er, -Er, Er, -Er]  # r
    ])

    def __init__(
            self,
            dt,
            max_rotor_speed=800, # rad/sec
            dtype=np.float32,
            name="SimpleMulticopter"
    ):
        super().__init__(dt, dtype=dtype, name=name)

        # action space
        self.act_low = np.zeros(self.NUM_ROTOR).astype(self.dtype)
        self.act_high = np.ones(self.NUM_ROTOR).astype(self.dtype) * float(max_rotor_speed)
        # observation space
        self.obs_low, self.obs_high = self.generate_inf_range(self.IX.sentinel)
        self.obs_low[:self.NUM_ROTOR] = 0
        self.obs_high[:self.NUM_ROTOR] = max_rotor_speed
        self.obs_low[self.IX.P:] = -xt.d2r(90.)
        self.obs_high[self.IX.P:] = xt.d2r(90.)

        self._x = np.zeros(self.IX.sentinel, dtype=dtype)
        self._x[self.IX.OMG] = self.omega0

        self._Ap = np.array([
            [1,  1, -1, -1],
            [1, -1, -1,  1],
            [0,  0,  0,  0]
        ], dtype=self.dtype) * self.L / 2
        self.kx = np.array([-1, -1, 1, 1], dtype=self.dtype) * self.L / np.sqrt(2)
        self.ky = np.array([1, -1, -1, 1], dtype=self.dtype) * self.L / np.sqrt(2)

    def __call__(self, action):
        action = action + self.omega0
        os = np.clip(action, self.act_low, self.act_high)

        sin = np.sin
        cos = np.cos
        tan = np.tan

        def df(x):
            u, v, w = x[self.IX.VEL]
            P, Q = x[self.IX.ANG]
            p, q = x[self.IX.ASP]

            # propeller speed
            # print("os:", os, ",  omega:", x[self.IX.OMG])
            do = (os - x[self.IX.OMG]) / self.tau
            Ts = 2.34E-5 * x[self.IX.OMG] ** 2 + 0.680

            # position
            du = -self.G * sin(Q) / self.M - w * q
            dv = self.G * sin(P) * cos(Q) / self.M + w * p
            dw = self.G * cos(P) * cos(Q) / self.M - np.sum(Ts) / self.M + u * q - v * p
            dus = np.array([du, dv, dw], dtype=self.dtype)
            Rinv = np.array([
                [cos(Q), sin(P) * sin(Q), cos(P) * sin(Q)],
                [0, cos(P), -1 * sin(P)],
                [-1 * sin(Q), sin(P) * cos(Q), cos(P) * cos(Q)]
            ], dtype=self.dtype)
            dXs = Rinv @ x[self.IX.VEL]

            # angle
            dp = self.kx @ Ts / self.Ixx
            dq = self.ky @ Ts / self.Iyy
            dps = np.array([dp, dq], dtype=self.dtype)

            Rangle = np.array([
                [1, sin(P) * tan(Q)],
                [0, cos(P)]
            ], dtype=self.dtype)
            dPs = Rangle @ x[self.IX.ASP]

            return np.concatenate([do, dXs, dus, dPs, dps])

        dx = xsim.no_time_rungekutta(df, self.dt, self._x)
        self._x += dx * self.dt

        return self._x.copy()

    def reset(self):
        self._x = np.zeros(self.IX.sentinel, dtype=self.dtype)
        self._x[self.IX.OMG] = self.omega0

    def get_state(self):
        return self._x.astype(self.dtype)

    def get_position(self):
        return self._x[self.IX.POS].astype(self.dtype)

    def get_angle(self):
        return self._x[self.IX.ANG].astype(self.dtype)

