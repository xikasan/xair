# coding: utf-8

import xsim
import pandas as pd
import matplotlib.pyplot as plt
from xair.envs.lvaircraft_random_pitch import LVAircraftPitchV4

DUE = 120
DT = 0.01


def run():
    env = LVAircraftPitchV4(DT)
    env.reset()

    log = xsim.Logger()

    for time in xsim.generate_step_time(DUE, DT):
        obs = env.observation
        act = env.action_space.sample()
        log.store(time=time, obs=obs).flush()
        env.step(act)

    res = xsim.Retriever(log)
    res = pd.DataFrame({
        "time": res.time(),
        "T": res.obs(env.IX_T),
        "C": res.obs(env.IX_C)
    })
    res.plot(x="time", y=["C", "T"])
    plt.show()

if __name__ == '__main__':
    run()
