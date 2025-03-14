import numpy as np

def init_standard_parameters():
	# samples, seq_len, s_rate, dim
	return 1000, 100, 24, 5

def sine_data_generation(waves=1000, samples=1e6, seq_len=2e2, cycles=2e5, dim=5):
	data = []

	if not samples/cycles >= 2: # ~ 5
		raise Exception("Less than 2 samples per cycle. Choose a different frequency or sampling rate.")

	f = cycles/seq_len # ~ 1000
	sd = 0.02*f
	
	for _ in range(waves):
        
		seq = []
		for _dim in range(dim):
			freq = np.random.normal(f, sd)
			phase = np.random.normal(f, sd)
			x = np.linspace(0, int(seq_len), int(samples))
			y = np.sin(freq*x*2*np.pi+phase)
			seq.append((y+1)*0.5)
		seq = np.transpose(np.array(seq))
		data.append(seq)
	data = np.array(data).astype(np.float32)
	return data
