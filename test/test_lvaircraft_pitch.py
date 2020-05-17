# coding: utf-8

import xsim
import xtools as xt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from xair.envs.lvaircraft_pitch import *


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


def test_v2():
    dt = 0.02
    due = 120

    env_tau = 0.5
    env_range = [-3, 3]
    env_period = 20.0

    # fail_mode = "GAIN_REDUCTION"
    fail_mode = "STABILITY_LOSS"

    env = LVAircraftPitchV2(
        dt=dt,
        tau=env_tau,
        range_target=xt.d2r(env_range),
        target_period=env_period
    )
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

        if env.get_time() == 60.0:
            env.set_fail_mode(fail_mode, val=-0.5)

    result = xsim.Retriever(log)
    result = pd.DataFrame({
        "time": result.time(),
        "reference": result.observation(env.IX_C),
        "pitch": result.observation(env.IX_T),
        "elevator": result.observation(env._model.IX_de)
    })
    result.plot(x="time", y=["reference", "pitch"])
    plt.show()


def test_v3():
    dt = 0.02
    due = 120

    env_tau = 0.5
    env_range = [-3, 3]
    env_period = 20.0

    # fail_mode = "GAIN_REDUCTION"
    fail_mode = "STABILITY_LOSS"
    fail_range = [-0.5, 0.5]
    num_round = 100

    env = LVAircraftPitchV3(
        dt=dt,
        tau=env_tau,
        range_target=xt.d2r(env_range),
        target_period=env_period,
        fail_mode=fail_mode,
        fail_range=fail_range
    )
    xt.info("env", env)

    Ks = np.array([2.5, 1.5, -2.5])

    log = []

    for r in range(num_round):

        log.append(xsim.Logger())
        env.reset()

        for time in xsim.generate_step_time(due, dt):
            obs = env.get_observation()
            act = np.array([0, Ks.dot(obs)], dtype=np.float32)
            log[-1].store(
                time=env.get_time(),
                observation=env.get_observation(),
                action=act
            ).flush()

            env.step(act)

            if env.get_time() == 60.0:
                env.set_fail()
            if (env.get_time() % 10) == 0:
                xt.info("round:{:2.0f} | time:{:3.0f}".format(r, env.get_time()))

    fig, ax = plt.subplots()
    for l in log:
        result = xsim.Retriever(l)
        result = pd.DataFrame({
            "time": result.time(),
            "reference": result.observation(env.IX_C),
            "pitch": result.observation(env.IX_T),
            "elevator": result.observation(env._model.IX_de)
        })
        result.plot(x="time", y="pitch", ax=ax, style="b", legend=False)

    result.plot(x="time", y="reference", ax=ax, style="r", legend=False)
    plt.show()


if __name__ == '__main__':
    # test_v0()
    # test_v1()
    # test_v2()
    test_v3()
