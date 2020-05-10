# coding: utf-8

import xsim
import xtools as xt
import pandas as pd
from xair.models.lvaircraft import LVAircraftEx
from matplotlib import pyplot as plt


def run():
    dt = 0.1
    due = 1000

    air = LVAircraftEx(dt)
    state = air.reset()

    logger = xsim.Logger()

    # initial log
    time = 0
    logger.store(time=time, state=state).flush()

    for time in xsim.generate_step_time(due, dt):
        action = xt.d2r([0.0, -0.01])
        state = air(action)

        logger.store(time=time, state=state).flush()

    result = xsim.Retriever(logger)
    result = pd.DataFrame({
        "time": result.time,
        "theta": xt.r2d(result.state(air.IX_q))
    })

    result.plot(x="time", y="theta")
    plt.show()


if __name__ == '__main__':
    run()
