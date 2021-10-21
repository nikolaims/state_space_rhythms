import pylab as plt
import numpy as np

from settings import FS


class SingleRhythmModel:
    def __init__(self, f, a, sigma, tau, fs=FS, x0=None):
        self.f = f
        self.a = a
        self.sigma = sigma
        self.tau = tau
        self.fs = fs

        self.x = x0 or np.zeros(2)
        self.y = None

        arg = 2*np.pi*f/fs
        self.F = self.a * np.array([[np.cos(arg), -np.sin(arg)],
                                    [np.sin(arg), np.cos(arg)]])

    def step(self):
        self.x = self.F @ self.x + np.random.randn(2) * self.sigma
        self.y = self.x[0] + np.random.randn() * self.tau
        return self.x, self.y

    def steps(self, n_steps):
        x = np.zeros((n_steps, 2))
        y = np.zeros(n_steps)
        for k in range(n_steps):
            x[k], y[k] = self.step()
        return x, y


if __name__ == '__main__':
    np.random.seed(42)

    n_steps = 3*FS
    t = np.arange(n_steps)/FS
    srm = SingleRhythmModel(10, 0.99, 0.1, 1)

    x, y = srm.steps(n_steps)
    plt.plot(t, y, 'C1')

    plt.plot(t, x[:, 0], 'C0')
    plt.plot(t, x[:, 1], '--C0')
    plt.plot(t, np.sum(x**2, 1)**0.5, '-C2')
    plt.show()