from rtlsdr import RtlSdr
import numpy as np
import scipy as sp

sdr = RtlSdr()

# Configure device
sdr.sample_rate = 2e6
sdr.center_freq = 1090e6
sdr.gain = 'auto'

print(sdr.valid_gains_db)

# ADSB Matched filter preamble
ADSB_preamble = np.array([1,0,1,0,0,0,0,1,0,1])

try:
    while True:
        samples = sdr.read_samples(512*1024)
        samples_abs = np.abs(samples)
        correlation = sp.signal.correlate(samples_abs, ADSB_preamble, mode='valid')
        threshold = np.max(correlation)
        print(f"samples_abs_max: {np.max(samples_abs)}")

except KeyboardInterrupt:
    print("Stopping.")
finally:
    sdr.close()