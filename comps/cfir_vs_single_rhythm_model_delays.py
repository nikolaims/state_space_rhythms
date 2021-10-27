from scipy.optimize import minimize

from single_rhytm.estimate import fun, get_result_params, inv_transform
from single_rhytm.kalman import SingleRhythmKalman, FixedLagKalman
from single_rhytm.log_likelihood import inv_sigmoid
from single_rhytm.model import SingleRhythmModel
from settings import FS
import pylab as plt
import scipy.signal as sg
import numpy as np
from cfir.env import CFIRBandDetector

np.random.seed(42)

nor = lambda x: (x - x.mean())/x.std()
srm = SingleRhythmModel(10, 0.995, 2, 20)
n_steps = 30 * FS
t = np.arange(n_steps)/FS

x, y = srm.steps(n_steps)
env_true = np.sum(x**2, 1)**0.5

x0 = inv_transform([10-2, 0.995-0.1, 2+1.5])
res = minimize(lambda _x: fun(_x, y.copy()), np.array(x0), method='BFGS', options={'disp': True}, tol=1)
res_params = get_result_params(res.x, y)

cfir_corrs = []
kalman_corrs = []
delays = [0, 1, 5, 10, 25, 50]
for delay in delays:
    cfir = CFIRBandDetector([8, 12], FS, delay)
    env_cfir = np.abs(cfir.apply(y))


    srk = SingleRhythmKalman(*res_params)
    if delay == 0:
        _, _, x_filter, V_filter = srk.collect_states(y)
    else:
        flk = FixedLagKalman(srk, delay)
        x_filter, V_filter = flk.collect_states(y)
    env_kalman = np.sum(x_filter**2, 1)**0.5

    cfir_corrs.append(np.corrcoef(env_true[:-delay or n_steps], env_cfir[delay:])[1,0])
    kalman_corrs.append(np.corrcoef(env_true[:-delay or n_steps], env_kalman[delay:])[1, 0])

plt.plot(delays, cfir_corrs)
plt.plot(delays, kalman_corrs)
