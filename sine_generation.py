import numpy as np

def init_standard_parameters():
	# samples, seq_len, s_rate, dim
	return 1000, 100, 1/24, 5

def sine_data_generation(n_samples=1000, seq_len=100, s_rate=1/24, dim=5, mean=5, sd=1):
	data = []

	if not 1/((mean + 3*sd) * 2) * 1/s_rate >= 1:
		raise Exception("Less than 2 samples per cycle. Choose a different frequency or sampling rate.")
	
	for _ in range(n_samples):
        
		seq = []
		for _dim in range(dim):
			freq = np.random.normal(mean, sd)
			phase = np.random.normal(mean, sd)
			x = np.linspace(0, seq_len*s_rate, seq_len)
			y = np.sin(freq*x*2*np.pi+phase)
			seq.append((y+1)*0.5)
		seq = np.transpose(np.array(seq))
		data.append(seq)
	data = np.array(data).astype(np.float32)
	return data
