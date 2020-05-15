# coding: utf-8

import xsim
import xtools as xt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from xair.envs.lvaircraft_pitch import LVAircraftPitchV0, LVAircraftPitchV1


def test_v0():
    dt = 0.1
    due = 10.0

    env = LVAircraftPitchV0(dt)
    xt.info("env", env)

    Ks = np.array([2.5, 1.5, -2.5])

    log = xsim.Logger()
    env.reset()

    for time in xsim.generate_step_time(due, dt):
        obs = env.get_observation()
        act = np.array([0, Ks.dot(obs)], dtype=np.float32)
        log.store(
            time=env.get_time(),
            observation=env.get_observation(),
            action=act
        ).flush()

        env.step(act)

    result = xsim.Retriever(log)
    result = pd.DataFrame({
        "time": result.time(),
        "reference": result.observation(env.IX_C),
        "pitch": result.observation(env.IX_T),
        "elevator": result.observation(env._model.IX_de)
    })
    result.plot(x="time", y=["reference", "pitch"])
    plt.show()


def test_v1():
    dt = 0.1
    due = 60.0

    env = LVAircraftPitchV1(dt, tau=0.2)
    xt.info("env", env)

    env.reset()
    xt.info("observation", env.get_observation())

    Ks = np.array([2.5, 1.5, -3.5])

    log = xsim.Logger()
    env.reset()

    for time in xsim.generate_step_time(due, dt):
        obs = env.get_observation()
        act = np.array([0, Ks.dot(obs)], dtype=np.float32)
        log.store(
            time=env.get_time(),
            observation=env.get_observation(),
            action=act
        ).flush()

        env.step(act)

    result = xsim.Retriever(log)
    result = pd.DataFrame({
        "time": result.time(),
        "reference": result.observation(env.IX_C),
        "pitch": result.observation(env.IX_T),
        "elevator": result.observation(env._model.IX_de)
    })
    result.plot(x="time", y=["reference", "pitch"])
    plt.show()


if __name__ == '__main__':
    # test_v0()
    test_v1()
