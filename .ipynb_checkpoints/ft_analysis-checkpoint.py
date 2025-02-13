# Fourier Transform Fidelity Analysis

import numpy as np
from scipy.fft import fft, fftfreq
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def fidelity_analysis(ori_data, no=1000, seq_len=100, s_rate=1/24, dim=5, unnorm=True, plot=False, tol=0.05):
	total = []
	
	for o in ori_data:
		data = o
		if unnorm:
			data = 2*np.transpose(o)-1
		subtotal = []
		for d in data:
			yf = np.abs(fft(d)) # transform data
			xf = fftfreq(seq_len, s_rate) # frequency spread, find analytic sample rate expression
			
			#subtotal.append(xf[np.where(np.isclose(yf, max(np.abs(yf))))][0]) # get frequency
			idx = np.argmax(yf)
			freq = xf[idx]
			subtotal.append(freq)
			
		total.append(np.mean(subtotal))

	stat, p = stats.shapiro(total)
	mean, sd = stats.norm.fit(total)

	if plot:
		plt.hist(total)
		plt.show()
		

	if p >= tol:
		print(f"{p} >= {tol} --- Likely Gaussian Distribution")
		print(f"The distribution has mean {mean} and standard deviation {sd}")
		return True
	else:
		print(f"{p} < {tol} --- Unlikely Gaussian Distribution")
		return False

