import sys
import numpy as np
import pyModeS as pms
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QLabel, QFrame
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

# --- WARNING: EXTERNAL DEPENDENCY ---
# The 'adi' library is required for the live streaming functionality.
try:
    import adi 
except ImportError:
    adi = None 

# Set pyqtgraph to use a dark theme for better contrast
pg.setConfigOption('background', 'k')
pg.setConfigOption('foreground', 'w')

# ======================================================================
# ADSB DECODING UTILITIES (Copied from User Input)
# ======================================================================

# Reference Position for position decoding (used by pyModeS)
REF_LAT = 40.936709
REF_LON = -73.97187

def adsb_message_decoder(data_block):
    """ Decodes a full ADSB message from bytes using pyModeS."""
    DATA_string = data_block.hex()
    
    DF = data_block[0] >> 3
    CA = data_block[0] & 0x07
    ME = data_block[4:12]
    TC = (ME[0] >> 3) & 0x1F

    if TC >= 1 and TC <= 4:
        # Aircraft Identification
        callsign = pms.adsb.callsign(DATA_string)
        return {"DF": DF, "CA": CA, "ICAO": data_block[1:4].hex().upper(), 
                "TypeCode": TC, "Callsign": callsign}
    elif (TC >= 5 and TC <= 8) or (TC >= 9 and TC <= 18) or (TC >= 20 and TC <= 22):
        # Position / Altitude related
        altitude = pms.adsb.altitude(DATA_string)
        
        # Position decoding requires two position messages (odd/even) and reference, 
        # but for a single message decode, we calculate based on reference.
        try:
            lat, lon = pms.adsb.position_with_ref(DATA_string, REF_LAT, REF_LON)
        except:
            lat, lon = None, None

        result = {"DF": DF, "CA": CA, "ICAO": data_block[1:4].hex().upper(), 
                  "TypeCode": TC, "Altitude": altitude}
        if lat and lon:
            result.update({"Latitude": lat, "Longitude": lon})
        return result
    elif TC == 19:
        # Airborne Velocity
        speed, heading, vertical_rate, _ = pms.adsb.velocity(DATA_string)
        return {"DF": DF, "CA": CA, "ICAO": data_block[1:4].hex().upper(), 
                "TypeCode": TC, "Speed": speed, "Heading": heading, "VerticalRate": vertical_rate}
    else:
        return {"DF": DF, "CA": CA, "ICAO": data_block[1:4].hex().upper(), 
                "TypeCode": TC, "Info": "Other or Reserved Message Type"}

def adsb_checksum_check(payload_bytes: bytes) -> bool:
    """ Checks the 24-bit CRC of a 112-bit ADS-B message."""
    if len(payload_bytes) != 14:
        return False
    message_int = int.from_bytes(payload_bytes, byteorder='big')
    ADSB_GENERATOR = 0b1111111111111010000001001
    remainder = message_int 
    
    for i in range(88):
        if (remainder & (1 << (111 - i))):
            shift = 87 - i
            remainder ^= (ADSB_GENERATOR << shift)
    
    # Check if the last 24 bits are zero
    return (remainder & 0xFFFFFF) == 0

def simple_adsb_data_block_decoder(samples_abs):
    """ Converts 224 magnitude samples (112 bits) into 14 bytes."""
    if len(samples_abs) != 224:
        return b''

    bit_string = "".join(
        "1" if samples_abs[i] > samples_abs[i+1] else "0"
        for i in range(0, 224, 2)
    )
    big_int = int(bit_string, 2)
    return big_int.to_bytes(14, byteorder='big')

def advanced_adsb_decoder(samples_abs : np.array, SNR_threshold: int = 4):
    """
    Scans for the 8us preamble pattern, checks SNR, decodes payload,
    and attempts single bit-flip correction on checksum failure.
    """
    messages = []
    # Only iterate up to where a full 14-byte message (224 samples) can start
    for i in range(samples_abs.size - 240): 
        # Check for basic preamble pattern (1,0,1,0,0,0,0,1,0,1,0,0,0,0) - 14 samples (7us)
        if (samples_abs[i+0] > samples_abs[i+1] and
            samples_abs[i+1] < samples_abs[i+2] and
            samples_abs[i+2] > samples_abs[i+3] and
            samples_abs[i+3] < samples_abs[i+0] and # Pattern check 1
            samples_abs[i+4] < samples_abs[i+0] and
            samples_abs[i+5] < samples_abs[i+0] and
            samples_abs[i+6] < samples_abs[i+0] and
            samples_abs[i+7] > samples_abs[i+8] and
            samples_abs[i+8] < samples_abs[i+9] and
            samples_abs[i+9] > samples_abs[i+6]): # Pattern check 2
            
            # Now check SNR
            high_avg = (samples_abs[i+0] + samples_abs[i+2] + samples_abs[i+7] + samples_abs[i+9]) / 4
            low_avg = (samples_abs[i+1] + samples_abs[i+3] + samples_abs[i+4] + samples_abs[i+5] + samples_abs[i+6] + samples_abs[i+8]) / 6
            SNR = high_avg / (low_avg + 1e-10)
            
            if SNR > SNR_threshold:
                # Payload starts at index i+16 (after the 8us preamble)
                data_block = simple_adsb_data_block_decoder(samples_abs[i+16:i+16+224])
                
                if adsb_checksum_check(data_block):
                    messages.append(data_block)
                else:
                    # Attempt single bit-flip correction
                    for bit_pos in range(112):
                        byte_index = bit_pos // 8
                        bit_index = 7 - (bit_pos % 8)
                        modified_block = bytearray(data_block)
                        modified_block[byte_index] ^= (1 << bit_index)
                        
                        if adsb_checksum_check(bytes(modified_block)):
                            messages.append(bytes(modified_block))
                            break
    return messages


# ======================================================================
# SDR PLOTTER CLASS (GUI and Live Streaming)
# ======================================================================

class SDRPlotter(QMainWindow):
    """
    PyQt6 application window to display SDR data and decoded ADS-B messages.
    """
    def __init__(self, sdr_device):
        super().__init__()
        self.setWindowTitle("PlutoSDR ADS-B Monitor")
        self.setGeometry(100, 100, 1400, 750) 
        self.sdr_device = sdr_device
        self.update_interval_ms = 50 
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout is a horizontal split (Plots | Messages)
        self.main_h_layout = QHBoxLayout(self.central_widget)
        
        self.init_plots()
        self.init_message_display()
        
        # Initialize and start the QTimer for live updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot_data)
        
        if self.sdr_device:
            # Get and display initial data before starting the timer
            try:
                initial_data = self.sdr_device.rx()[0]
                self.plot_sdr_data(initial_data)
                self.timer.start(self.update_interval_ms)
            except Exception as e:
                self.message_display.append(f"ERROR: Initial RX failed: {e}. Check PlutoSDR connection.")
                if self.sdr_device:
                    self.sdr_device.rx_destroy_buffer()


    def init_plots(self):
        # Left side: All plots stacked vertically
        plot_container = QWidget()
        plot_v_layout = QVBoxLayout(plot_container)
        self.plot_items = {}
        
        # --- 1. Time Domain Plot (I/Q) ---
        self.time_plot = pg.PlotWidget(title="1. Time Domain: I (Real) and Q (Imaginary)")
        self.time_plot.setLabel('bottom', "Sample Index (Time)")
        self.time_plot.setLabel('left', "Amplitude")
        self.time_plot.addLegend()
        self.time_plot.showGrid(x=True, y=True, alpha=0.5)
        self.time_plot.setXRange(0, 500) 
        self.plot_items['i_data'] = self.time_plot.plot([], pen='b', name='In-phase (I)')
        self.plot_items['q_data'] = self.time_plot.plot([], pen='r', name='Quadrature (Q)')
        
        # --- 2. Absolute Magnitude Plot ---
        self.magnitude_plot = pg.PlotWidget(title="2. Time Domain: Absolute Magnitude (|I+jQ|)")
        self.magnitude_plot.setLabel('bottom', "Sample Index (Time)")
        self.magnitude_plot.setLabel('left', "Magnitude")
        self.magnitude_plot.showGrid(x=True, y=True, alpha=0.5)
        self.magnitude_plot.setXRange(0, 500) 
        self.plot_items['mag_data'] = self.magnitude_plot.plot([], pen='c', name='|I+jQ|')

        # --- 3. Frequency Spectrum Plot (FFT) ---
        self.fft_plot = pg.PlotWidget(title="3. Frequency Spectrum (FFT)")
        self.fft_plot.setLabel('bottom', "Normalized Frequency (f/Fs)")
        self.fft_plot.setLabel('left', "Power Density (dB)")
        self.fft_plot.showGrid(x=True, y=True, alpha=0.5)
        self.fft_plot.setXRange(-0.5, 0.5)
        self.plot_items['fft_data'] = self.fft_plot.plot([], pen='y')

        # Add plots to the vertical layout
        plot_v_layout.addWidget(self.time_plot)
        plot_v_layout.addWidget(self.magnitude_plot)
        plot_v_layout.addWidget(self.fft_plot)
        
        # Add the plot container to the main horizontal layout (left side)
        self.main_h_layout.addWidget(plot_container, 2) # Plots take 2/3 of the width

    def init_message_display(self):
        # Right side: Message Log
        message_container = QFrame()
        message_container.setFrameShape(QFrame.Shape.StyledPanel)
        message_v_layout = QVBoxLayout(message_container)
        
        title = QLabel("Real-Time ADS-B Message Log")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; padding-bottom: 5px;")
        message_v_layout.addWidget(title)
        
        self.message_display = QTextEdit()
        self.message_display.setReadOnly(True)
        self.message_display.setStyleSheet("font-family: monospace; font-size: 10pt; background-color: #222; color: #0F0;")
        message_v_layout.addWidget(self.message_display)
        
        # Add the message container to the main horizontal layout (right side)
        self.main_h_layout.addWidget(message_container, 1) # Messages take 1/3 of the width


    def plot_sdr_data(self, data):
        """Calculates and updates all plot items with new data using setData."""
        if data is None or len(data) == 0:
            return

        N = len(data)
        t = np.arange(N) 
        
        # --- Time Domain Updates ---
        i_component = np.real(data)
        q_component = np.imag(data)
        magnitude = np.abs(data)
        
        self.plot_items['i_data'].setData(t, i_component)
        self.plot_items['q_data'].setData(t, q_component)
        self.plot_items['mag_data'].setData(t, magnitude)

        # --- Frequency Spectrum Updates ---
        fft_result = np.fft.fft(data)
        fft_shifted = np.fft.fftshift(fft_result)
        psd_db = 10 * np.log10(np.abs(fft_shifted)**2 / N + 1e-15) 
        
        fs = 1.0 
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

        self.plot_items['fft_data'].setData(freqs, psd_db)

    def update_plot_data(self):
        """Called periodically by the QTimer to fetch and plot new data and decode messages."""
        if not self.sdr_device:
            self.timer.stop()
            return
            
        try:
            # 1. Read new samples from the SDR
            samples = self.sdr_device.rx()[0] 
            
            # 2. Update the plots
            self.plot_sdr_data(samples)
            
            # 3. Perform ADS-B Decoding
            magnitude = np.abs(samples)
            decoded_msgs = advanced_adsb_decoder(magnitude, SNR_threshold=4)
            print(f"Decoded {len(decoded_msgs)} ADS-B messages in this frame.")
            # 4. Display Decoded Messages
            for data_block in decoded_msgs:
                decoded_info = adsb_message_decoder(data_block)
                if decoded_info:
                    # Construct clean output string
                    icao = decoded_info.get('ICAO', 'N/A')
                    tc = decoded_info.get('TypeCode', 'N/A')
                    
                    output_parts = [f"ICAO: {icao} (TC {tc})"]
                    
                    if 'Callsign' in decoded_info:
                        output_parts.append(f"Callsign: {decoded_info['Callsign'].strip()}")
                    if 'Altitude' in decoded_info:
                        output_parts.append(f"Alt: {decoded_info['Altitude']} ft")
                    if 'Latitude' in decoded_info:
                        output_parts.append(f"Pos: ({decoded_info['Latitude']:.4f}, {decoded_info['Longitude']:.4f})")
                    if 'Speed' in decoded_info:
                        output_parts.append(f"Speed: {decoded_info['Speed']} kts")
                        
                    output_str = " | ".join(output_parts)
                    
                    # Append to the QTextEdit log
                    self.message_display.append(output_str)

        except Exception as e:
            # Catch exceptions like connection loss
            print(f"Error during data acquisition: {e}. Stopping stream.")
            self.message_display.append(f"--- STREAM ERROR: {e} ---")
            self.timer.stop()


# ----------------------------------------------------------------------
# Main execution function
# ----------------------------------------------------------------------
def main():
    if adi is None:
        print("Fatal Error: 'adi' library (Analog Devices) is required but not installed.")
        sys.exit(1)

    # 1. Initialize SDR
    try:
        sdr = adi.ad9361(uri="ip:192.168.1.38")
    except Exception as e:
        print(f"Failed to initialize SDR device: {e}")
        print("Check if the PlutoSDR is powered on and accessible at 192.168.1.38.")
        sys.exit(1)

    # 2. Set SDR Parameters
    sdr.sample_rate = int(2_000_000)
    sdr.rx_lo = int(1090_000_000)
    sdr.rx_rf_bandwidth = int(2_000_000)
    sdr.rx_buffer_size = 65536 
    sdr.gain_control_mode_chan0 = 'slow_attack'

    print(f"SDR initialized. LO: {sdr.rx_lo / 1e6} MHz, Sample Rate: {sdr.sample_rate / 1e6} MSPS.")
    
    # Clear Buffer
    try:
        # Blocks until buffer is full
        _ = sdr.rx()
    except Exception as e:
        print(f"Initial buffer clear failed: {e}")
        sdr.rx_destroy_buffer()
        sys.exit(1)


    # 3. Start PyQt Application
    app = QApplication(sys.argv)
    window = SDRPlotter(sdr)
    window.show()
    
    # 4. Cleanup when the application closes
    app.aboutToQuit.connect(lambda: sdr.rx_destroy_buffer())
    
    # Start the event loop
    sys.exit(app.exec())

if __name__ == '__main__':
    main()