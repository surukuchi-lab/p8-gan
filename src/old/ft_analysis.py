# Fourier Transform Fidelity Analysis

import numpy as np
from scipy.fft import fft, fftfreq
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def fidelity_analysis(ori_data, seq_len=2e2, samples=1e6, cycles=2e5, unnorm=True, plot=False, tol=0.05):
	total = []
	
	for o in range(len(ori_data)):
		if o % int(0.1*len(ori_data)) == 0:
			print(f"Completed {o/len(ori_data)*100:.2f}%")
			
		data = ori_data[o]
		if unnorm:
			data = 2*np.transpose(ori_data[o])-1
		subtotal = []
		for d in data:
			yf = np.abs(fft(d)) # transform data
			xf = fftfreq(int(samples), seq_len/samples) # frequency spread, find analytic sample rate expression
			
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

