from scipy import signal
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 4, 1, 3, 5])

dt = 0.01  # time step in seconds (e.g., 100 Hz sampling rate)

corr = signal.correlate(a, b, mode='full')
lags = signal.correlation_lags(len(a), len(b), mode='full')

time_lags = lags * dt  # convert to time

best_lag_time = time_lags[np.argmax(corr)]
print("Best lag (samples):", lags[np.argmax(corr)])
print("Best lag (seconds):", best_lag_time)

