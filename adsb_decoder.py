from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as sp_signal
import matplotlib.pyplot as plt # Import for plotting

# --- Configuration ---
sdr = RtlSdr()

# Configure device
sdr.sample_rate = 2e6
sdr.center_freq = 1090e6
sdr.gain = 49.6 # Using 'auto' gain
SAMPLES_PER_READ = 5120 * 1024 # Defines the size of the array read

# Decoding threshold for binary decoding
DECODE_THRESHOLD = 0.1 

# ADSB Matched filter preamble 
ADSB_preamble = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

try:
    print(f"Starting single SDR read (Center Freq: {sdr.center_freq/1e6} MHz, Sample Rate: {sdr.sample_rate/1e6} Msps)")

    # Read complex samples from the SDR
    samples = sdr.read_samples(SAMPLES_PER_READ)

    SAMPLES_TO_SKIP = 2000
    samples = samples[SAMPLES_TO_SKIP:]

    # Calculate the absolute magnitude of the complex samples (signal envelope)
    samples_abs = np.abs(samples)

    # 1. Correlation Plot Data
    # Perform the correlation (matched filtering for preamble)
    # I am definitly not doing this right...
    correlation = sp_signal.correlate(samples_abs, ADSB_preamble, mode='valid')
    max_correlation = np.max(correlation)
    
    # Determine the Y limits for the correlation plot to keep it static
    corr_min = np.min(correlation)
    corr_max = np.max(correlation)
    # Add a 5% buffer above and below the min/max values
    y2_min = corr_min * 0.95 if corr_min < 0 else corr_min * 0.95
    y2_max = corr_max * 1.05

    # 2. Bit Decode Plot Data (Highly simplified: just checks if magnitude is above threshold)
    # Convert samples_abs to 0 or 1 based on the simple threshold
    decoded_bits = (samples_abs > DECODE_THRESHOLD).astype(int)

    print(f"Read {len(samples_abs)} processed samples. Max Correlation Peak: {max_correlation:.2f}.")



    # --- Plotting Logic: Linking X-Axes ---
    # Create the figure and two subplots, sharing the x-axis (sharex=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    plt.suptitle(f"ADSB Signal Analysis @ 1090 MHz (X-Axes Linked, Y-Axes Static)", fontsize=16)

    # Subplot 1 (ax1): Decoded Bits (0 or 1)
    # Plotting ALL available samples. Y-limits are already fixed.
    ax1.plot(decoded_bits, drawstyle='steps-post', color='darkgreen', label='Threshold Decoded Bits')
    ax1.plot(samples_abs, alpha=0.3, color='grey', label='Raw Magnitude (I/Q)')
    ax1.axhline(DECODE_THRESHOLD, color='r', linestyle='--', linewidth=1, label='Decode Threshold')
    
    ax1.set_title(f'Simplified Binary Decoding ({len(decoded_bits)} Total Samples)')
    ax1.set_ylabel('Magnitude / Decoded Value')
    ax1.set_ylim(-0.1, 1.5) # Fixed Y-limits for decoding plot
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y')
    
    # Subplot 2 (ax2): Correlation Graph
    ax2.plot(correlation, color='darkblue', label='Correlation Output')
    ax2.axhline(y=max_correlation, color='r', linestyle='-', linewidth=2, label=f'Max Correlation ({max_correlation:.2f})')
    
    ax2.set_title('Correlation of Signal Magnitude with ADSB Preamble')
    ax2.set_xlabel('Sample Index / Correlation Lag (Linked)')
    ax2.set_ylabel('Correlation Value')
    ax2.set_ylim(y2_min, y2_max) # Fixed Y-limits for correlation plot
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show() # Display the plots

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Always close the SDR device when done
    print("Closing SDR device.")
    sdr.close()
