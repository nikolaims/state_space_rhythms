from scipy.optimize import minimize
from single_rhytm.model import SingleRhythmModel
from single_rhytm.kalman import SingleRhythmKalman
from single_rhytm.log_likelihood import nll_and_tau, sigmoid, inv_sigmoid
from settings import FS

import numpy as np


def transform(params):
    f = sigmoid(params[0]) * FS / 2
    a = sigmoid(params[1])
    sigma = 10. ** params[2]
    return np.array([f, a, sigma])


def inv_transform(params):
    f = inv_sigmoid(params[0] / FS * 2)
    a = inv_sigmoid(params[1])
    sigma = np.log10(params[2])
    return np.array([f, a, sigma])


def fun(params, y):
    f, a, sigma = transform(params)
    srk = SingleRhythmKalman(f, a, sigma, 1)
    x_pred, V_pred, x_filter, V_filter = srk.collect_states(y)
    nll, tau = nll_and_tau(x_pred, V_pred, y, srk.H)
    return nll

def get_result_params(x, y):
    f, a, sigma = transform(x)
    srk = SingleRhythmKalman(f, a, sigma, 1)
    x_pred, V_pred, x_filter, V_filter = srk.collect_states(y)
    nll, tau = nll_and_tau(x_pred, V_pred, y, srk.H)
    return f, a, sigma*tau, tau


if __name__ == '__main__':
    srm = SingleRhythmModel(10, 0.99, 0.1, 2)
    n_steps = 5 * FS
    x, y = srm.steps(n_steps)


    x0 = inv_transform([1, 0.9, 1])

    res = minimize(lambda x: fun(x, y), np.array(x0), method='BFGS', options={'disp': True}, tol=1)
    res_x = get_result_params(res.x, y)


    srk = SingleRhythmKalman(*res_x)
    _x_pred, _V_pred, x_filter, V_filter = srk.collect_states(y)
    t = np.arange(n_steps) / FS

    import pylab as plt
    plt.plot(t, x_filter[:, 0])
    plt.fill_between(t, x_filter[:, 0] - 2 * V_filter[:, 0, 0] ** 0.5, x_filter[:, 0] + 2 * V_filter[:, 0, 0] ** 0.5,
                     alpha=0.5)
    plt.plot(t, x[:, 0])