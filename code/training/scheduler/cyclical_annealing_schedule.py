import numpy as np
import math


def frange_cycle_linear(start, stop, n_steps, n_cycle=4, ratio=0.5):
    L = np.ones(n_steps)
    period = n_steps / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_steps):
            L[int(i + c * period)] = v
            v += step
            i += 1

    return L


def frange_cycle_sigmoid(start, stop, n_steps, n_cycle=4, ratio=0.5):
    L = np.ones(n_steps)
    period = n_steps / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
            v += step
            i += 1
    return L


def frange_cycle_cosine(start, stop, n_steps, n_cycle=4, ratio=0.5):
    L = np.ones(n_steps)
    period = n_steps / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [0, pi] for plots:

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 0.5 - .5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L


"""
Linear scheduler.

start: value to start "beta" (e.g. 0)
stop: value to stop increasing "beta" (e.g. 1)
step: how much increase "beta" at each iteration
n_steps: number of total iterations
"""


def frange(start, stop, step, n_steps):
    L = np.ones(n_steps)
    v, i = start, 0
    while v <= stop:
        L[i] = v
        v += step
        i += 1
    return L


class BetaScheduler:

    def __init__(self, b_min=0., b_max=1.):

        self.min = b_min
        self.max = b_max

        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self._step_count = -1
        self.step()

    def get_beta(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self):
        self._step_count += 1
        value = self.get_beta()
        self.beta = value


class ConstantBeta(BetaScheduler):

    def __int__(self, b_min=0., b_max=1.):
        super(self, ConstantBeta).__init__( b_min, b_max)

    def get_beta(self):
        return self.max


class CycleLinearBeta(BetaScheduler):

    def __init__(self, b_min=0., b_max=1., steps=25, n_cycle=4, ratio=0.5):
        self.max_steps = steps
        self.n_cycle = n_cycle
        self.ratio = ratio

        self.values = frange_cycle_linear(b_min, b_max, steps, n_cycle, ratio)


        super().__init__( b_min, b_max)

    def get_beta(self):
        if self._step_count >= self.max_steps:
            return self.max

        return self.values[self._step_count]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.plot(frange(0, 1, 0.1, 50))
    plt.show()

    plt.plot(frange_cycle_cosine(0, 1, 50, 3, 0.5))
    plt.show()

    plt.plot(frange_cycle_sigmoid(0, 1, 50, 3, 0.5))
    plt.show()

    plt.plot(frange_cycle_linear(0, 1, 50, 3, 0.5))
    plt.show()

    # get scheduler
    beta = 0.0

    scheduler = CycleLinearBeta()

    for i in range(100):


        scheduler.step()
        print(scheduler.beta)
