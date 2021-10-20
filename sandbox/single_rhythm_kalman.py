import pylab as plt
import numpy as np

from settings import FS


class SingleRhythmKalman:
    def __init__(self, f, a, sigma, tau, fs=FS):
        self.f = f
        self.a = a
        self.sigma = sigma
        self.tau = tau
        self.fs = fs

        self.x_n_nm1 = np.zeros(2)
        self.V_n_nm1 = np.eye(2)
        self.x_n_n = np.zeros(2)
        self.V_n_n = np.eye(2)

        arg = 2*np.pi*f/fs
        self.F = self.a * np.array([[np.cos(arg), -np.sin(arg)],
                                    [np.sin(arg), np.cos(arg)]])
        self.H = np.array([1, 0])

    def filter(self, y):
        K = self.V_n_nm1 @ self.H / (self.H @ self.V_n_nm1 @ self.H + self.tau**2)
        self.x_n_n = self.x_n_nm1 + K * (y - self.H @ self.x_n_nm1)
        self.V_n_n = self.V_n_nm1 - K.reshape(-1, 1) @ self.H.reshape(1, -1) @ self.V_n_nm1
        return self.x_n_n, self.V_n_nm1

    def pred(self):
        self.x_n_nm1 = self.F @ self.x_n_n
        self.V_n_nm1 = self.F @ self.V_n_n @ self.F.T + self.sigma**2

from sandbox.single_rhythm_model import SingleRhythmModel
np.random.seed(42)
srm = SingleRhythmModel(10, 0.99, 0.1, 2)
srk = SingleRhythmKalman(10, 0.99, 0.1, 2)

n_steps = FS
x, y = srm.steps(n_steps)
x_pred = np.zeros((n_steps, 2))
V_pred = np.zeros((n_steps, 2, 2))
for n in range(n_steps):
    srk.pred()
    x_pred[n], V_pred[n] = srk.filter(y[n])

t = np.arange(n_steps)/FS

plt.plot(t, x_pred[:, 0])
plt.fill_between(t, x_pred[:, 0]-2*V_pred[:, 0, 0]**0.5, x_pred[:, 0]+2*V_pred[:, 0, 0]**0.5, alpha=0.5)
plt.plot(t, x[:, 0])