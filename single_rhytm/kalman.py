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

        self.x_n_nm1 = None
        self.V_n_nm1 = None
        self.x_n_n = None
        self.V_n_n = None

        arg = 2*np.pi*f/fs
        self.F = self.a * np.array([[np.cos(arg), -np.sin(arg)],
                                    [np.sin(arg), np.cos(arg)]])
        self.H = np.array([1, 0])

        self.first_step = True

    def filter(self, y):
        K = self.V_n_nm1 @ self.H / (self.H @ self.V_n_nm1 @ self.H + self.tau**2)
        self.x_n_n = self.x_n_nm1 + K * (y - self.H @ self.x_n_nm1)
        self.V_n_n = self.V_n_nm1 - K.reshape(-1, 1) @ self.H.reshape(1, -1) @ self.V_n_nm1
        return self.x_n_n, self.V_n_n

    def pred(self):
        if not self.first_step:
            self.x_n_nm1 = self.F @ self.x_n_n
            self.V_n_nm1 = self.F @ self.V_n_n @ self.F.T + self.sigma**2
        else:
            self.x_n_nm1 = self.initial_x()
            self.V_n_nm1 = self.initial_v()
            self.first_step = False
        return self.x_n_nm1, self.V_n_nm1

    def initial_x(self):
        return np.zeros(2)

    def initial_v(self):
        return np.eye(2) * self.sigma ** 2 / (1 - self.a ** 2)


    def collect_states(self, y):
        n_steps = len(y)
        x_pred = np.zeros((n_steps, 2))
        V_pred = np.zeros((n_steps, 2, 2))
        x_filter = np.zeros((n_steps, 2))
        V_filter = np.zeros((n_steps, 2, 2))
        for n in range(n_steps):
            x_pred[n], V_pred[n] = self.pred()
            x_filter[n], V_filter[n] = self.filter(y[n])
        return x_pred, V_pred, x_filter, V_filter


class FixedLagKalman:
    def __init__(self, kalman, N):
        self.kalman = kalman
        self.H = self.kalman.H
        self.tau = self.kalman.tau
        self.sigma = self.kalman.sigma
        self.F = self.kalman.F

        self.N = N

        self.first_step = True

    def smooth_step(self, y):
        if self.first_step:
            x0 = self.kalman.initial_x()
            S0 = self.kalman.initial_v()
            self.x_k_km1 = [x0] + [np.zeros_like(x0)]*(self.N+1)
            self.S_k_km1 = [S0] + [np.zeros_like(S0)]*(self.N+1)
            self.first_step = False

        _const1 = self.H / (self.H @ self.S_k_km1[0] @ self.H + self.tau**2)
        K_k_0 = self.S_k_km1[0] @ _const1
        L_k_0 = self.F @ K_k_0

        _const2 = self.F - L_k_0 @ self.H
        x_kp1_k_0 = _const2 @ self.x_k_km1[0] + L_k_0 * y
        S_kp1_k_0 = self.F @ self.S_k_km1[0] @ _const2.T + self.sigma**2
        self.x_kp1_k = [x_kp1_k_0] + [None] * (self.N + 1)
        self.S_kp1_k = [S_kp1_k_0] + [None] * (self.N + 1)

        S_diag = self.S_k_km1[0].copy()

        for i in range(1, self.N+1 + 1):
            L = self.S_k_km1[i-1] @ _const1
            self.S_kp1_k[i] = self.S_k_km1[i-1] @ _const2.T
            self.x_kp1_k[i] = self.x_k_km1[i-1] + L * (y - self.F @ self.x_k_km1[0])
            S_diag = S_diag - (self.S_k_km1[i-1].T @ self.H) @ L.T

        self.x_k_km1 = self.x_kp1_k
        self.S_k_km1 = self.S_kp1_k
        return self.x_kp1_k[-1], S_diag

    def collect_states(self, y):
        n_steps = len(y)
        x_filter = np.zeros((n_steps, 2))
        V_filter = np.zeros((n_steps, 2, 2))
        for n in range(n_steps):
            x_filter[n], V_filter[n] = self.smooth_step(y[n])
        return x_filter, V_filter

if __name__ == '__main__':
    from single_rhytm.model import SingleRhythmModel
    np.random.seed(42)
    srm = SingleRhythmModel(10, 0.99, 0.1, 2)
    srk = SingleRhythmKalman(10, 0.99, 0.1, 2)

    n_steps = FS*3
    x, y = srm.steps(n_steps)
    _x_pred, _V_pred , x_filter, V_filter = srk.collect_states(y)


    t = np.arange(n_steps)/FS

    plt.plot(t, x_filter[:, 0])
    plt.plot(t, (x_filter**2).sum(1)**0.5, '--C0')
    plt.fill_between(t, x_filter[:, 0]-2*V_filter[:, 0, 0]**0.5, x_filter[:, 0]+2*V_filter[:, 0, 0]**0.5, alpha=0.5)
    plt.plot(t, x[:, 0])
    plt.plot(t, (x ** 2).sum(1) ** 0.5, '--C1')

    # tau_hat = ((y - srk.H @ x_pred)**2/(srk.H.repeat(500).reshape(2, 500) @ V_pred @ srk.H + 1)).sum()

    plt.figure()
    lag = int(0.1*FS)
    flk = FixedLagKalman(srk, lag)
    x_flk, V_flk = flk.collect_states(y)

    plt.plot(t, x_flk[:, 0])
    plt.plot(t, (x_flk ** 2).sum(1) ** 0.5, '--C0')
    plt.fill_between(t, x_flk[:, 0] - 2 * V_flk[:, 0, 0] ** 0.5, x_flk[:, 0] + 2 * V_flk[:, 0, 0] ** 0.5,
                     alpha=0.5)
    plt.plot(t[:-lag], np.roll(x[:, 0], lag)[:-lag])
    plt.plot(t[:-lag], (np.roll(x, lag, axis=0)[:-lag] ** 2).sum(1) ** 0.5, '--C1')