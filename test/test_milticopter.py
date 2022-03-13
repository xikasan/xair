# coding: utf-8

import xsim
import xtools as xt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xair.models.multicopter import SimpleMulticopter
from xair.envs.multicopter_position import MulticopterPosition2DV0
from matplotlib import pyplot as plt


due = 30.
dt = 0.01

def run():
    model = SimpleMulticopter(dt)
    model.reset()
    print("act:", model.act_high)
    print("obs:", model.obs_high)

    omegas_0 = np.ones(model.NUM_ROTOR) * model.omega0 #* 600
    log = xsim.Logger()
    print(model.get_state())

    domega = 0.
    past_w = 0.
    K = 0.

    for time in xsim.generate_step_time(due, dt):
        state = model.get_state()
        w = state[model.IX.W]
        dw = (w - past_w) / dt
        action = np.zeros(model.NUM_ROTOR)

        log.store(time=time, state=state, action=action, dw=dw, domega=domega).flush()

        model(action)
        past_w = w

    print("steady omega:", action)
    print("steady state:", model.get_state()[model.IX.OMG])
    # print(log.buffer())
    # exit()
    ret = xsim.Retriever(log)
    res = pd.DataFrame(dict(
        time=ret.time(),
        z=ret.state(model.IX.Z),
        w=ret.state(model.IX.W),
        dw=ret.dw(),
        domega_cmd=ret.action(0),
        omega=ret.state(model.IX.o1)
    ))

    fig, axes = plt.subplots(nrows=5, sharex=True)
    res.plot(x="time", y="z", ax=axes[0])
    res.plot(x="time", y="w", ax=axes[1])
    res.plot(x="time", y="dw", ax=axes[2])
    res.plot(x="time", y="omega", ax=axes[3])
    res.plot(x="time", y="domega_cmd", ax=axes[4])
    plt.savefig("docs/temp.png")


def run_Q():
    model = SimpleMulticopter(dt)
    model.reset()
    Einv = np.linalg.inv(model.E)

    Q_target = xt.d2r(1.)
    log = xsim.Logger()

    for time in xsim.generate_step_time(due, dt):
        state = model.get_state()
        Q = state[model.IX.Q]
        q = state[model.IX.q]
        Q_error = Q_target - Q
        dq_cmd = 200. * Q_error - 100. * q
        omega_cmd = Einv @ np.array([0, 0, dq_cmd, 0]) * 1e-5

        log.store(time=time, Q=Q, Q_target=Q_target, dq_cmd=dq_cmd, omega_cmd=omega_cmd).flush()

        model(omega_cmd)

    ret = xsim.Retriever(log)
    res = pd.DataFrame(dict(
        time=ret.time(),
        Q=xt.r2d(ret.Q()),
        Q_target=xt.r2d(ret.Q_target()),
        dq_cmd=xt.r2d(ret.dq_cmd()),
        omega_cmd_1=ret.omega_cmd(0),
        omega_cmd_2=ret.omega_cmd(1),
        omega_cmd_3=ret.omega_cmd(2),
        omega_cmd_4=ret.omega_cmd(3),
    ))

    fig, axes = plt.subplots(nrows=3, sharex=True)
    res.plot(x="time", y=["Q_target", "Q"], ax=axes[0])
    res.plot(x="time", y="dq_cmd", ax=axes[1])
    res.plot(x="time", y=["omega_cmd_1", "omega_cmd_2", "omega_cmd_3", "omega_cmd_4"], ax=axes[2])
    plt.savefig("docs/temp2.png")


def run_x():
    model = SimpleMulticopter(dt)
    model.reset()
    Einv = np.linalg.inv(model.E)
    max_Q = 10.

    x_target = 1.
    log = xsim.Logger()

    ix = model.IX

    for time in xsim.generate_step_time(120, dt):
        state = model.get_state()
        x = state[ix.X]
        u = state[ix.U]
        Q = state[ix.Q]
        q = state[ix.q]

        x_error = x_target - x
        Q_target_ = xt.d2r(- 0.2 * x_error + 2. * u)
        Q_target = np.clip(Q_target_, -xt.d2r(max_Q), xt.d2r(max_Q))
        Q_error = Q_target - Q
        dq_cmd = 100. * Q_error - 100. * q
        omega_cmd = Einv @ np.array([0, 0, dq_cmd, 0]) * 1e-5

        log.store(time=time, state=state, target=[x_target, Q_target], omega_cmd=omega_cmd).flush()

        model(omega_cmd)

    ret = xsim.Retriever(log)
    res = pd.DataFrame(dict(
        time=ret.time(),
        x=ret.state(ix.X),
        Q=ret.state(ix.Q),
        x_target=ret.target(0),
        Q_target=ret.target(1)
    ))

    fig, axes = plt.subplots(nrows=2, sharex=True)
    res.plot(x="time", y=["x_target", "x"], ax=axes[0])
    res.plot(x="time", y=["Q_target", "Q"], ax=axes[1])
    plt.savefig("docs/temp_x.png")


def run_env():
    env = MulticopterPosition2DV0()
    env._get_reward()


if __name__ == '__main__':
    run_env()
