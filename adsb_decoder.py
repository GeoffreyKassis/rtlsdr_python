from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as sp_signal
import matplotlib.pyplot as plt # Import for plotting
import pyModeS as pms



# --- Configuration ---
sdr = RtlSdr()

# Configure device
sdr.sample_rate = 2e6
sdr.center_freq = 1090e6
sdr.gain = 49.6 # Using 'auto' gain
SAMPLES_PER_READ = 512 * 1024 # Defines the size of the array read

# Decoding threshold for binary decoding
DECODE_THRESHOLD = 0.1 

# New vibe coded method (Zero-Mean)
# This is a more standard approach for matched filtering
ADSB_preamble = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
mean_preamble = np.mean(ADSB_preamble)
zero_mean_preamble = ADSB_preamble - mean_preamble

# Ok this works way better, I think I understand it.
# Basically before normailizing any power meant the correlation was high, now its looking for the specific shape of the power

# Reference Position
REF_LAT = 40.936709
REF_LON = -73.97187

def adsb_message_decoder(data_block):
    """ Decodes a full ADSB message from a bytes.
    
    Args:
        data_block: bytes array of length 14 (112 bits)
    Returns:
        ADSB: Dict containing decoded fields
    """
    DATA_string = data_block.hex()
    
    DF = data_block[0] >> 3
    CA = data_block[0] & 0x07
    ICAO = data_block[1:4]
    ME = data_block[4:12]
    TC = (ME[0] >> 3) & 0x1F
    PI = data_block[12:14]

    print(f"Decoded ADSB Message: DF={DF}, CA={CA}, ICAO={ICAO.hex().upper()}, TC={TC}")


    if TC >= 1 and TC <= 4:
        # Aircraft Identification
        callsign = pms.adsb.callsign(DATA_string)
        return {
            "DF": DF,
            "CA": CA,
            "ICAO": ICAO.hex().upper(),
            "TypeCode": TC,
            "Callsign": callsign
        }
    elif TC >= 5 and TC <= 8:
        # Surface Position
        altitude = pms.adsb.altitude(DATA_string)
        return {
            "DF": DF,
            "CA": CA,
            "ICAO": ICAO.hex().upper(),
            "TypeCode": TC,
            "Altitude": altitude
        }
    elif TC >= 9 and TC <= 18:
        # Airborne Position (w/Baro Altitude)
        altitude = pms.adsb.altitude(DATA_string)
        lat, lon = pms.adsb.position_with_ref(DATA_string, REF_LAT, REF_LON)  # No reference position
        return {
            "DF": DF,
            "CA": CA,
            "ICAO": ICAO.hex().upper(),
            "TypeCode": TC,
            "Altitude": altitude,
            "Latitude": lat,
            "Longitude": lon
        }
    elif TC == 19:
        # Airborne Velocity
        speed, heading, vertical_rate, speed_type = pms.adsb.velocity(DATA_string)
        return {
            "DF": DF,
            "CA": CA,
            "ICAO": ICAO.hex().upper(),
            "TypeCode": TC,
            "Speed": speed,
            "Heading": heading,
            "VerticalRate": vertical_rate
        }
    elif TC >= 20 and TC <= 22:
        # Airborne Position (w/GPS Altitude)
        altitude = pms.adsb.altitude(DATA_string)
        lat, lon = pms.adsb.position_with_ref(DATA_string, REF_LAT, REF_LON)  # No reference position
        return {
            "DF": DF,
            "CA": CA,
            "ICAO": ICAO.hex().upper(),
            "TypeCode": TC,
            "Altitude": altitude,
            "Latitude": lat,
            "Longitude": lon
        }
    elif TC >= 23 and TC <= 27:
        # Reserved for future use
        return {
            "DF": DF,
            "CA": CA,
            "ICAO": ICAO.hex().upper(),
            "TypeCode": TC,
            "Info": "Reserved for future use"
        }
    elif TC == 28:
        # Comm-B Altitude Reply
        altitude = pms.adsb.altitude(DATA_string)
        return {
            "DF": DF,
            "CA": CA,
            "ICAO": ICAO.hex().upper(),
            "TypeCode": TC,
            "Altitude": altitude
        }
    elif TC == 29:
        # Target state and status information
        return {
            "DF": DF,
            "CA": CA,
            "ICAO": ICAO.hex().upper(),
            "TypeCode": TC,
            "Info": "Target state and status information"
        }
    else:
        return 

def adsb_checksum_check(payload_bytes: bytes) -> bool:
    """
    Checks the 24-bit CRC of a 112-bit ADS-B message.
    Input is a 14-byte 'bytes' object.
    This version uses efficient integer-only math.
    """
    if len(payload_bytes) != 14:
        raise ValueError(f"Expected 14 bytes, but got {len(payload_bytes)}")

    # Convert 14-byte (112-bit) message into a single integer
    message_int = int.from_bytes(payload_bytes, byteorder='big')
    
    # 25-bit generator polynomial as an integer
    ADSB_GENERATOR = 0b1111111111111010000001001

    # This is the "remainder" we will be modifying.
    remainder = message_int 
    
    # We iterate 88 times, for the 88 bits of data
    for i in range(88):
        # We check the bit from the left (MSB, bit 111) down to bit 24
        # (111 - i) will go from 111 down to 24
        if (remainder & (1 << (111 - i))):
            # If the bit is set, XOR with the generator
            # We align the 25-bit generator to the bit we are checking
            # The shift is (112 - 25 - i) = (87 - i)
            shift = 87 - i
            remainder ^= (ADSB_GENERATOR << shift)
    
    # After 88 loops, the first 88 bits are zero.
    # The last 24 bits (bits 23 down to 0) contain the CRC remainder.
    # If the message was valid, this remainder should be zero.
    
    # 0xFFFFFF is the mask for the last 24 bits
    if (remainder & 0xFFFFFF) == 0:
        print("--------------Checksum Successful------------")
        return True
    else:
        print(f"Checksum Failed. Remainder: {remainder & 0xFFFFFF:06X}")
        return False

def simple_adsb_data_block_decoder(samples_abs):
    """
    Simple ADSB Payload decoder.
    Input is 224 samples of absolute magnitude.
    Returns the decoded 112 bits as 14 bytes.
    """
    # An ADS-B data block is 112 bits (14 bytes), encoded in 224 samples.
    if len(samples_abs) != 224:
        raise ValueError(f"Expected 224 samples for a 112-bit block, but got {len(samples_abs)}")

    # 1. Decode all 112 bits into a string representation
    # We use a list comprehension and then ''.join() for efficiency
    bit_string = "".join(
        "1" if samples_abs[i] > samples_abs[i+1] else "0"
        for i in range(0, 224, 2)
    )

    # 2. Convert the 112-bit string into a single large integer
    big_int = int(bit_string, 2)

    # 3. Convert the integer into a 14-byte object (112 bits = 14 bytes)
    # 'big' byteorder means the most significant bit (bit_string[0])
    # becomes the most significant bit of the first byte.
    return big_int.to_bytes(14, byteorder='big')

def simple_adsb_decoder(samples_abs : np.array, SNR_threshold: int = 4):
    """Simple ADSB Decoder 
    
    Inspiried by dump1090
    1) Scan for preamble pattern in the magnitude samples
    2) Check SNR of premable
    3) If preamble detected, decode the next 224 samples as ADSB message
    4) Check checksum of decoded message

    Args:
        samples_abs: numpy array of samples from SDR (magnitude)
        SNR_threshold: SNR threshold for preamble detection, default 4
    Returns:
        messages: List of ADSB messages as byte arrays
    """
    # Test variable for amount of false positives
    false_positives = 0
    positives = 0
    messages = []
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
                data_block = simple_adsb_data_block_decoder(samples_abs[i+16:i+224+16])
                if adsb_checksum_check(data_block):
                    messages.append(data_block)
            else:
                false_positives += 1
    print(f"Total false positives: {false_positives}")
    print(f"Total positive rate: {(positives/false_positives)*100}%")
    return messages

try:
    print(f"Starting single SDR read (Center Freq: {sdr.center_freq/1e6} MHz, Sample Rate: {sdr.sample_rate/1e6} Msps)")
    while True:
        samples = sdr.read_samples(SAMPLES_PER_READ)
        SAMPLES_TO_SKIP = 2000
        samples = samples[SAMPLES_TO_SKIP:]
        samples_abs = np.abs(samples)

        # UPDATED: Now simple_adsb_decoder returns the list of objects AND the indices
        # messages is a list of ADSBMessage objects
        messages = simple_adsb_decoder(samples_abs) 
        
        print("\n--- Summary of Decoded Messages ---")
        for msg in messages:
            print(adsb_message_decoder(msg))


#     # 2. Bit Decode Plot Data (Highly simplified: just checks if magnitude is above threshold)
#     decoded_bits = (samples_abs > DECODE_THRESHOLD).astype(int)
#     print(f"Read {len(samples_abs)} processed samples. Max Correlation Peak: {max_correlation:.2f}.")


# # --- Plotting Logic: Linking X-Axes ---
#     # Create the figure and two subplots, sharing the x-axis (sharex=True)
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
#     plt.suptitle(f"ADSB Signal Analysis @ 1090 MHz (Total Valid Messages: {len(messages)})", fontsize=16)

#     # Subplot 1 (ax1): Decoded Bits (0 or 1)
#     ax1.plot(samples_abs, alpha=0.8, color='purple', label='Raw Magnitude (I/Q)', 
#              linestyle='-', drawstyle='steps-post', markersize=4) 
#     ax1.axhline(DECODE_THRESHOLD, color='r', linestyle='--', linewidth=1, label='Decode Threshold')
    
#     # Plot vertical lines for detected indices on the magnitude plot (ax1)
#     for i in indicies:
#         label = 'Preamble Detected' if i == indicies[0] else None 
#         ax1.axvline(x=i, color='darkorange', linestyle='-', linewidth=1.5, label=label, alpha=0.7)

#     ax1.set_title(f'Simplified Binary Decoding ({len(samples_abs)} Total Samples)')
#     ax1.set_ylabel('Magnitude / Decoded Value')
#     ax1.set_ylim(-0.1, 1.5)
#     ax1.legend(loc='upper right')
#     ax1.grid(True, axis='y')
    
#     # Subplot 2 (ax2): Correlation Graph
#     correlation = sp_signal.correlate(samples_abs, zero_mean_preamble, mode='valid')
#     max_correlation = np.max(correlation)
#     corr_min = np.min(correlation)
#     corr_max = np.max(correlation)
#     y2_min = corr_min * 0.95 if corr_min < 0 else corr_min * 0.95
#     y2_max = corr_max * 1.05

#     ax2.plot(correlation, color='darkblue', label='Correlation Output', 
#              linestyle='-', linewidth=1, markersize=4)
#     ax2.axhline(y=max_correlation, color='r', linestyle='-', linewidth=2, label=f'Max Correlation ({max_correlation:.2f})')
    
#     # Plot vertical lines for detected indices on the correlation plot (ax2)
#     for i in indicies:
#         label = 'Preamble Detected' if i == indicies[0] else None 
#         ax2.axvline(x=i, color='darkorange', linestyle='-', linewidth=1.5, label=label, alpha=0.7)

#     ax2.set_title('Correlation of Signal Magnitude with ADSB Preamble')
#     ax2.set_xlabel('Sample Index / Correlation Lag (Linked)')
#     ax2.set_ylabel('Correlation Value')
#     ax2.set_ylim(y2_min, y2_max)
#     ax2.legend(loc='upper right')
#     ax2.grid(True)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

except Exception as e:
    # Use standard error logging
    raise e
    print(f"An error occurred: {e}")
    # Re-raise the exception if you want the program to halt with a stack trace
    # raise 
finally:
    # Always close the SDR device when done
    print("Closing SDR device.")
    sdr.close()