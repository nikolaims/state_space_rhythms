from scipy.optimize import minimize

from single_rhytm.estimate import fun, get_result_params, inv_transform
from single_rhytm.kalman import SingleRhythmKalman
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
n_steps = 60 * FS
t = np.arange(n_steps)/FS

x, y = srm.steps(n_steps)
env_true = np.sum(x**2, 1)**0.5


cfir = CFIRBandDetector([8, 12], FS, 0)
env_cfir = np.abs(cfir.apply(y))

x0 = inv_transform([10-2, 0.995-0.1, 2+1.5])
res = minimize(lambda _x: fun(_x, y.copy()), np.array(x0), method='BFGS', options={'disp': True}, tol=1)
res_params = get_result_params(res.x, y)


srk = SingleRhythmKalman(*res_params)
_x_pred, _V_pred, x_filter, V_filter = srk.collect_states(y)
env_kalman = np.sum(x_filter**2, 1)**0.5



plt.figure()
axes = [plt.subplot(221), plt.subplot(222)]
axes[0].plot(t, x[:, 0])
axes[0].plot(t, y, zorder=-2)
axes[1].semilogy(*sg.welch(x[:, 0], FS, nperseg=2*FS, noverlap=2*FS-FS//2, nfft=4*FS), label='x_1')
axes[1].semilogy(*sg.welch(y, FS, nperseg=2*FS, noverlap=2*FS-FS//2, nfft=4*FS), label='y')
axes[1].set_xlabel('freq')
axes[1].set_xlim(0, 30)
axes[1].legend()


ax = plt.subplot(223)

ax.plot(t, nor(env_true), label='true 1')
ax.plot(t, nor(env_cfir), label=f'cfir {np.corrcoef(env_true, env_cfir)[1,0]:.3f}')
ax.plot(t, nor(env_kalman), label=f'kalman {np.corrcoef(env_true, env_kalman)[1,0]:.3f}')
ax.set_xlabel('time, s')

ax.legend()

corrs_cfir = [np.corrcoef(env_true[:-k or len(x)], env_cfir[k:])[1,0] for k in range(FS//2)]
corrs_kalman = [np.corrcoef(env_true[:-k or len(x)], env_kalman[k:])[1,0] for k in range(FS//2)]

ax = plt.subplot(224)
t = np.arange(FS//2)/FS
ind_cfir = np.argmax(corrs_cfir)
ind_kalman = np.argmax(corrs_kalman)
ax.plot(t, corrs_cfir, 'C1', label='cfir')
ax.axvline(t[ind_cfir], color='C1')
ax.axhline(corrs_cfir[ind_cfir], color='C1')
ax.plot(t, corrs_kalman, 'C2', label='kalman')
ax.axvline(t[ind_kalman], color='C2')
ax.axhline(corrs_kalman[ind_kalman], color='C2')
ax.set_xlabel('delay, s')
ax.set_ylabel('corr. coef., s')
ax.legend()
ax.grid()

plt.subplots_adjust(wspace=0.5, hspace=0.5)
