from single_rhytm.model import SingleRhythmModel
from single_rhytm.kalman import SingleRhythmKalman
import numpy as np
from settings import FS



def sigmoid(x):
    return 1/(1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x) - np.log(1-x)

def nll_and_tau(x_pred, V_pred, y, H):
    HVH_p1 = (H.reshape(1, -1) @ V_pred @ H + 1)[:, 0]
    tau2 = ((y - x_pred @ H) ** 2 / HVH_p1).mean()
    L = len(y)/2*(np.log(2*np.pi*tau2) + 1) + 0.5 * np.sum(np.log(HVH_p1))
    return L, tau2**0.5


if __name__ == '__main__':
    np.random.seed(42)
    srm = SingleRhythmModel(10, 0.99, 0.1, 2)
    # srk = SingleRhythmKalman(10, 0.99, 0.1, 2)

    n_steps = 10 * FS
    x, y = srm.steps(n_steps)

    srk = SingleRhythmKalman(10, 0.99, 0.1/2, 1)
    x_pred, V_pred, x_filter, V_filter = srk.collect_states(y)
    print(nll_and_tau(x_pred, V_pred, y, srk.H))

    f_list = np.arange(0, 20, 2)
    f_nll_list = np.zeros(len(f_list))
    f_tau_list = np.zeros(len(f_list))
    for k, f in enumerate(f_list):
        srk = SingleRhythmKalman(f, srm.a, srm.sigma / srm.tau, 1)
        x_pred, V_pred, x_filter, V_filter = srk.collect_states(y)
        f_nll_list[k], f_tau_list[k] = nll_and_tau(x_pred, V_pred, y, srk.H)

    a_list = sigmoid(np.arange(1, 10))
    a_nll_list = np.zeros(len(a_list))
    a_tau_list = np.zeros(len(a_list))
    for k, a in enumerate(a_list):
        srk = SingleRhythmKalman(srm.f, a, srm.sigma / srm.tau, 1)
        x_pred, V_pred, x_filter, V_filter = srk.collect_states(y)
        a_nll_list[k], a_tau_list[k] = nll_and_tau(x_pred, V_pred, y, srk.H)

    sigma_list = 10.**np.arange(-7, 1)
    sigma_nll_list = np.zeros(len(sigma_list))
    sigma_tau_list = np.zeros(len(sigma_list))
    for k, sigma in enumerate(sigma_list):
        srk = SingleRhythmKalman(srm.f, srm.a, sigma / srm.tau, 1)
        x_pred, V_pred, x_filter, V_filter = srk.collect_states(y)
        sigma_nll_list[k], sigma_tau_list[k] = nll_and_tau(x_pred, V_pred, y, srk.H)

    import pylab as plt
    fig, axes = plt.subplots(2, 3, sharey='row', sharex='col')
    axes[0, 0].plot(f_list, f_nll_list, label='f')
    axes[0, 0].axvline(srm.f, color='C1', label='f*')
    axes[1, 0].plot(f_list, f_tau_list, label='f')
    axes[1, 0].axvline(srm.f, color='C1', label='f*')
    axes[1, 0].axhline(srm.tau, color='C1', label='f*')
    axes[1, 0].set_xlabel('f')
    axes[0, 0].set_ylabel('Neg. log. likelihood')
    axes[1, 0].set_ylabel('tau')

    axes[0, 1].plot(inv_sigmoid(a_list), a_nll_list, label='a')
    axes[0, 1].axvline(inv_sigmoid(srm.a), color='C1', label='a*')
    axes[1, 1].plot(inv_sigmoid(a_list), a_tau_list, label='a')
    axes[1, 1].axvline(inv_sigmoid(srm.a), color='C1', label='a*')
    axes[1, 1].axhline(srm.tau, color='C1', label='a*')
    axes[1, 1].set_xlabel('inv_sigm(a)')

    axes[0, 2].plot(np.log10(sigma_list), sigma_nll_list, label='sigma')
    axes[0, 2].axvline(np.log10(srm.sigma), color='C1', label='sigma*')
    axes[1, 2].plot(np.log10(sigma_list), sigma_tau_list, label='sigma')
    axes[1, 2].axvline(np.log10(srm.sigma), color='C1', label='sigma*')
    axes[1, 2].axhline(srm.tau, color='C1', label='sigma*')
    axes[1, 2].set_xlabel('log10(sigma)')