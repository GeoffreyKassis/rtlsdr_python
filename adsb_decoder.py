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

# New vibe coded method (Zero-Mean)
# This is a more standard approach for matched filtering
ADSB_preamble = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
mean_preamble = np.mean(ADSB_preamble)
zero_mean_preamble = ADSB_preamble - mean_preamble

# Ok this works way better, I think I understand it.
# Basically before normailizing any power meant the correlation was high, now its looking for the specific shape of the power

def adsb_checksum_check(decoded_bits):
    print(len(decoded_bits))
    ADSB_GENERATOR = 0b1111111111111010000001001
    ADSB_GENERATOR = [int(bit) for bit in bin(ADSB_GENERATOR)[2:]]
    
    for i in range(0, 88):
        if decoded_bits[i] == 1:
            for j in range(0, 24):
                decoded_bits[i + j] = decoded_bits[i + j] ^ ADSB_GENERATOR[j]

    CRC = decoded_bits[-24:]

    if any(CRC):
        print("Checksum Failed: ", CRC)
        return False
    print("--------------Checksum Successful------------")
    return True

def simple_adsb_data_block_decoder(samples_abs):
    """ Simple ADSB Payload decoder. Input is 224 samples of absolute magnitude."""
    decoded_bits = []
    for i in range(0, len(samples_abs), 2):
        decoded_bits.append(1 if samples_abs[i] > samples_abs[i+1] else 0)
    return decoded_bits

def simple_adsb_decoder(samples):
    """ Simple premable detector similar to what dump1090 uses"""

    SNR_threshold = 4  # SNR threshold. Convert from dB to ratio

    # Calculate the absolute magnitude of the complex samples (signal envelope)
    samples_abs = np.abs(samples)

    # Test variable for amount of false positives
    false_positives = 0
    positives = 0
    indices = []
    for i in range(samples_abs.size - 10):
        # Check for basic preamble pattern. This uses relative power levels pretty cool
        if (samples_abs[i+0] > samples_abs[i+1] and
            samples_abs[i+1] < samples_abs[i+2] and
            samples_abs[i+2] > samples_abs[i+3] and
            samples_abs[i+3] < samples_abs[i+0] and
            samples_abs[i+4] < samples_abs[i+0] and
            samples_abs[i+5] < samples_abs[i+0] and
            samples_abs[i+6] < samples_abs[i+0] and
            samples_abs[i+7] > samples_abs[i+8] and
            samples_abs[i+8] < samples_abs[i+9] and
            samples_abs[i+9] > samples_abs[i+6]):
            
            # Now check that the low bits are significantly lower than the high bits
            high_avg = (samples_abs[i+0] + samples_abs[i+2] + samples_abs[i+7] + samples_abs[i+9]) / 4
            low_avg = (samples_abs[i+1] + samples_abs[i+3] + samples_abs[i+4] + samples_abs[i+5] + samples_abs[i+6] + samples_abs[i+8]) / 6
            # Calculate SNR
            SNR = high_avg / (low_avg + 1e-10)  # Avoid division by zero
            if  SNR > SNR_threshold:
                print(f"Preamble detected at index {i} with SNR: {SNR:.2f}")
                positives += 1
                indices.append(i)
                data_block = simple_adsb_data_block_decoder(samples_abs[i+16:i+224+16])
                print(f"Decoded Bits: {data_block}")
                adsb_checksum_check(data_block)
            else:
                false_positives += 1
    print(f"Total false positives: {false_positives}")
    print(f"Total positive rate: {(positives/false_positives)*100}%")
    return indices

try:
    print(f"Starting single SDR read (Center Freq: {sdr.center_freq/1e6} MHz, Sample Rate: {sdr.sample_rate/1e6} Msps)")

    # Read complex samples from the SDR
    samples = sdr.read_samples(SAMPLES_PER_READ)

    SAMPLES_TO_SKIP = 2000
    samples = samples[SAMPLES_TO_SKIP:]

    indicies = simple_adsb_decoder(samples)

    # Calculate the absolute magnitude of the complex samples (signal envelope)
    samples_abs = np.abs(samples)

    # 1. Correlation Plot Data
    # Perform the correlation (matched filtering for preamble)
    # Numpy correlation normalizes by default
    correlation = sp_signal.correlate(samples_abs, zero_mean_preamble, mode='valid')
    
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
    ax1.plot(samples_abs, alpha=0.8, color='purple', label='Raw Magnitude (I/Q)')
    ax1.axhline(DECODE_THRESHOLD, color='r', linestyle='--', linewidth=1, label='Decode Threshold')
    
    # >>> MODIFICATION START - Plot indices on ax1
    # Plot vertical lines for detected indices on the magnitude plot (ax1)
    for i in indicies:
        # The first one is labeled, the rest are not to avoid clutter in the legend
        label = 'Preamble Detected' if i == indicies[0] else None 
        ax1.axvline(x=i, color='darkorange', linestyle='-', linewidth=1.5, label=label, alpha=0.7)
    # >>> MODIFICATION END

    ax1.set_title(f'Simplified Binary Decoding ({len(decoded_bits)} Total Samples)')
    ax1.set_ylabel('Magnitude / Decoded Value')
    ax1.set_ylim(-0.1, 1.5) # Fixed Y-limits for decoding plot
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y')
    
    # Subplot 2 (ax2): Correlation Graph
    # Note: Correlation output is shorter than samples_abs by len(zero_mean_preamble) - 1. 
    # This is handled because we are using 'valid' mode, and the indices are for the raw samples_abs, 
    # but because the x-axes are linked, the visual placement is correct on the correlation plot.
    ax2.plot(correlation, color='darkblue', label='Correlation Output')
    ax2.axhline(y=max_correlation, color='r', linestyle='-', linewidth=2, label=f'Max Correlation ({max_correlation:.2f})')
    
    # >>> MODIFICATION START - Plot indices on ax2
    # Plot vertical lines for detected indices on the correlation plot (ax2)
    for i in indicies:
        # The first one is labeled, the rest are not to avoid clutter in the legend
        label = 'Preamble Detected' if i == indicies[0] else None 
        ax2.axvline(x=i, color='darkorange', linestyle='-', linewidth=1.5, label=label, alpha=0.7)
    # >>> MODIFICATION END

    ax2.set_title('Correlation of Signal Magnitude with ADSB Preamble')
    ax2.set_xlabel('Sample Index / Correlation Lag (Linked)')
    ax2.set_ylabel('Correlation Value')
    ax2.set_ylim(y2_min, y2_max) # Fixed Y-limits for correlation plot
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show() # Display the plots

except Exception as e:
    raise e
    print(f"An error occurred: {e}")
finally:
    # Always close the SDR device when done
    print("Closing SDR device.")
    sdr.close()
