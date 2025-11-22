from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox
import pyqtgraph as pg
import numpy as np
import time
import signal

# Defaults
fft_size = 4096
num_rows = 200
center_freq = 1090e6  # Changed to FM band for testing
sample_rates = [2.4, 2.0, 1.0, 0.5]  # MHz
sample_rate = sample_rates[0] * 1e6
time_plot_samples = 500
gain = 30  # Changed to 30 dB (RTL-SDR range is 0-50)

sdr_type = "rtl"

# Init SDR
if sdr_type == "pluto":
    import adi
    sdr = adi.Pluto("ip:192.168.1.10")
    sdr.rx_lo = int(center_freq)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(sample_rate*0.8)
    sdr.rx_buffer_size = int(fft_size)
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = gain
elif sdr_type == "rtl":
    import rtlsdr
    sdr = rtlsdr.RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = gain
elif sdr_type == "usrp":
    import uhd
    usrp = uhd.usrp.MultiUSRP(args="addr=192.168.1.201")
    usrp.set_rx_rate(sample_rate, 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
    usrp.set_rx_gain(gain, 0)
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    metadata = uhd.types.RXMetadata()
    streamer = usrp.get_rx_stream(st_args)
    recv_buffer = np.zeros((1, fft_size), dtype=np.complex64)
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    def flush_buffer():
        for _ in range(10):
            streamer.recv(recv_buffer, metadata)

class SDRWorker(QObject):
    def __init__(self):
        super().__init__()
        self.gain = gain
        self.sample_rate = sample_rate
        self.freq = int(center_freq / 1e3)  # in kHz
        self.spectrogram = -50*np.ones((fft_size, num_rows))
        self.PSD_avg = -50*np.ones(fft_size)
        
        # UI settings (not yet applied)
        self.ui_freq = self.freq
        self.ui_gain = self.gain
        self.ui_sample_rate_idx = 0
        
        self.is_paused = False
        self.settings_changed = False

    # PyQt Signals
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal()
    settings_applied = pyqtSignal()  # Signal when settings are done applying

    # PyQt Slots - just update UI values, don't touch SDR
    def update_freq_ui(self, val):
        self.ui_freq = val
        self.settings_changed = True

    def update_gain_ui(self, val):
        self.ui_gain = val
        self.settings_changed = True

    def update_sample_rate_ui(self, val):
        self.ui_sample_rate_idx = val
        self.settings_changed = True

    def apply_settings(self):
        """Pause, apply settings, resume - called from GUI thread"""
        print("Apply button pressed - pausing SDR...")
        self.is_paused = True
        time.sleep(1)
        QTimer.singleShot(100, self._do_apply_settings)  # Give current read time to finish

    def _do_apply_settings(self):
        """Actually apply the settings - runs in worker thread context via timer"""
        print("Applying settings to SDR...")
        
        if sdr_type == "pluto":
            if self.ui_freq != self.freq:
                sdr.rx_lo = int(self.ui_freq * 1e3)
                self.freq = self.ui_freq
            if self.ui_gain != self.gain:
                sdr.rx_hardwaregain_chan0 = self.ui_gain
                self.gain = self.ui_gain
            if sample_rates[self.ui_sample_rate_idx] * 1e6 != self.sample_rate:
                self.sample_rate = sample_rates[self.ui_sample_rate_idx] * 1e6
                sdr.sample_rate = int(self.sample_rate)
                sdr.rx_rf_bandwidth = int(self.sample_rate * 0.8)
                
        elif sdr_type == "rtl":
            try:
                if self.ui_freq != self.freq:
                    print(f"Setting frequency to {self.ui_freq} kHz")
                    sdr.center_freq = self.ui_freq * 1e3
                    self.freq = self.ui_freq
                    
                if self.ui_gain != self.gain:
                    print(f"Setting gain to {self.ui_gain} dB")
                    sdr.gain = self.ui_gain
                    self.gain = self.ui_gain
                    
                if sample_rates[self.ui_sample_rate_idx] * 1e6 != self.sample_rate:
                    new_rate = sample_rates[self.ui_sample_rate_idx] * 1e6
                    print(f"Setting sample rate to {new_rate/1e6} MHz")
                    sdr.sample_rate = new_rate
                    self.sample_rate = new_rate
                    
                # Wait for hardware to settle
                time.sleep(0.2)
                
                # Flush a few samples
                for _ in range(3):
                    try:
                        _ = sdr.read_samples(fft_size)
                    except:
                        pass
                        
            except Exception as e:
                print(f"Error applying settings: {e}")
                
        elif sdr_type == "usrp":
            if self.ui_freq != self.freq:
                usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.ui_freq * 1e3), 0)
                self.freq = self.ui_freq
            if self.ui_gain != self.gain:
                usrp.set_rx_gain(self.ui_gain, 0)
                self.gain = self.ui_gain
            if sample_rates[self.ui_sample_rate_idx] * 1e6 != self.sample_rate:
                self.sample_rate = sample_rates[self.ui_sample_rate_idx] * 1e6
                usrp.set_rx_rate(self.sample_rate, 0)
            flush_buffer()

        self.settings_changed = False
        print("Settings applied - resuming SDR")
        self.is_paused = False
        self.settings_applied.emit()

    # Main loop
    def run(self):
        start_t = time.time()

        # If paused, don't read from SDR
        if self.is_paused:
            time.sleep(0.01)
            self.end_of_run.emit()
            return

        try:
            if sdr_type == "pluto":
                samples = sdr.rx()/2**11
            elif sdr_type == "rtl":
                samples = sdr.read_samples(fft_size)
            elif sdr_type == "usrp":
                streamer.recv(recv_buffer, metadata)
                samples = recv_buffer[0]
            elif sdr_type == "sim":
                tone = np.exp(2j*np.pi*self.sample_rate*0.1*np.arange(fft_size)/self.sample_rate)
                noise = np.random.randn(fft_size) + 1j*np.random.randn(fft_size)
                samples = self.gain*tone*0.02 + 0.1*noise
                np.clip(samples.real, -1, 1, out=samples.real)
                np.clip(samples.imag, -1, 1, out=samples.imag)

            self.time_plot_update.emit(samples[0:time_plot_samples])

            PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/fft_size)
            self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
            self.freq_plot_update.emit(self.PSD_avg)

            self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1)
            self.spectrogram[:,0] = PSD
            self.waterfall_plot_update.emit(self.spectrogram)

            #print("Frames per second:", 1/(time.time() - start_t))
        except Exception as e:
            print(f"Error in worker run: {e}")
            time.sleep(1)
        
        self.end_of_run.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("The PySDR Spectrum Analyzer")
        self.setFixedSize(QSize(1500, 1000))

        self.spectrogram_min = 0
        self.spectrogram_max = 0

        layout = QGridLayout()

        # Initialize worker and thread
        self.sdr_thread = QThread()
        self.sdr_thread.setObjectName('SDR_Thread')
        worker = SDRWorker()
        worker.moveToThread(self.sdr_thread)

        # Time plot
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
        time_plot.setMouseEnabled(x=False, y=True)
        time_plot.setYRange(-1.1, 1.1)
        time_plot_curve_i = time_plot.plot([])
        time_plot_curve_q = time_plot.plot([])
        layout.addWidget(time_plot, 1, 0)

        # Time plot auto range buttons
        time_plot_auto_range_layout = QVBoxLayout()
        layout.addLayout(time_plot_auto_range_layout, 1, 1)
        auto_range_button = QPushButton('Auto Range')
        auto_range_button.clicked.connect(lambda : time_plot.autoRange())
        time_plot_auto_range_layout.addWidget(auto_range_button)
        auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
        auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
        time_plot_auto_range_layout.addWidget(auto_range_button2)

        # Freq plot
        freq_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'})
        freq_plot.setMouseEnabled(x=False, y=True)
        freq_plot_curve = freq_plot.plot([])
        freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
        freq_plot.setYRange(-30, 20)
        layout.addWidget(freq_plot, 2, 0)

        # Freq auto range button
        auto_range_button = QPushButton('Auto Range')
        auto_range_button.clicked.connect(lambda : freq_plot.autoRange())
        layout.addWidget(auto_range_button, 2, 1)

        # Layout container for waterfall related stuff
        waterfall_layout = QHBoxLayout()
        layout.addLayout(waterfall_layout, 3, 0)

        # Waterfall plot
        waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'})
        imageitem = pg.ImageItem(axisOrder='col-major')
        waterfall.addItem(imageitem)
        waterfall.setMouseEnabled(x=False, y=False)
        waterfall_layout.addWidget(waterfall)

        # Colorbar for waterfall
        colorbar = pg.HistogramLUTWidget()
        colorbar.setImageItem(imageitem)
        colorbar.item.gradient.loadPreset('viridis')
        imageitem.setLevels((-30, 20))
        waterfall_layout.addWidget(colorbar)

        # Waterfall auto range button
        auto_range_button = QPushButton('Auto Range\n(-2σ to +2σ)')
        def update_colormap():
            imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
            colorbar.setLevels(self.spectrogram_min, self.spectrogram_max)
        auto_range_button.clicked.connect(update_colormap)
        layout.addWidget(auto_range_button, 3, 1)

        # Freq slider with label, all units in kHz
        freq_slider = QSlider(Qt.Orientation.Horizontal)
        freq_slider.setRange(0, int(1.766e6))
        freq_slider.setValue(int(center_freq/1e3))
        freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        freq_slider.setTickInterval(int(1e6))
        freq_slider.valueChanged.connect(worker.update_freq_ui)  # Changed to valueChanged for live update
        freq_label = QLabel()
        def update_freq_label(val):
            freq_label.setText("Frequency [MHz]: " + str(val/1e3))
        freq_slider.valueChanged.connect(update_freq_label)
        update_freq_label(freq_slider.value())
        layout.addWidget(freq_slider, 4, 0)
        layout.addWidget(freq_label, 4, 1)

        # Gain slider with label
        gain_slider = QSlider(Qt.Orientation.Horizontal)
        gain_slider.setRange(0, 50)
        gain_slider.setValue(gain)
        gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        gain_slider.setTickInterval(5)
        gain_slider.valueChanged.connect(worker.update_gain_ui)
        gain_label = QLabel()
        def update_gain_label(val):
            gain_label.setText("Gain: " + str(val) + " dB")
        gain_slider.valueChanged.connect(update_gain_label)
        update_gain_label(gain_slider.value())
        layout.addWidget(gain_slider, 5, 0)
        layout.addWidget(gain_label, 5, 1)

        # Sample rate dropdown using QComboBox
        sample_rate_combobox = QComboBox()
        sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
        sample_rate_combobox.setCurrentIndex(0)
        sample_rate_combobox.currentIndexChanged.connect(worker.update_sample_rate_ui)
        sample_rate_label = QLabel()
        def update_sample_rate_label(val):
            sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
        sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
        update_sample_rate_label(sample_rate_combobox.currentIndex())
        layout.addWidget(sample_rate_combobox, 6, 0)
        layout.addWidget(sample_rate_label, 6, 1)

        # APPLY BUTTON - New!
        apply_button = QPushButton('APPLY SETTINGS')
        apply_button.setStyleSheet("QPushButton { font-weight: bold; background-color: #4CAF50; color: white; padding: 10px; }")
        apply_button.clicked.connect(worker.apply_settings)
        
        # Update button appearance based on whether settings changed
        def update_apply_button():
            if worker.settings_changed:
                apply_button.setStyleSheet("QPushButton { font-weight: bold; background-color: #FF9800; color: white; padding: 10px; }")
                apply_button.setText('APPLY SETTINGS *')
            else:
                apply_button.setStyleSheet("QPushButton { font-weight: bold; background-color: #4CAF50; color: white; padding: 10px; }")
                apply_button.setText('APPLY SETTINGS')
        
        def on_settings_applied():
            update_apply_button()
            freq_plot.autoRange()
        
        worker.settings_applied.connect(on_settings_applied)
        freq_slider.valueChanged.connect(lambda: update_apply_button())
        gain_slider.valueChanged.connect(lambda: update_apply_button())
        sample_rate_combobox.currentIndexChanged.connect(lambda: update_apply_button())
        
        layout.addWidget(apply_button, 7, 0, 1, 2)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Signals and slots stuff
        def time_plot_callback(samples):
            time_plot_curve_i.setData(samples.real)
            time_plot_curve_q.setData(samples.imag)

        def freq_plot_callback(PSD_avg):
            f = np.linspace(worker.freq*1e3 - worker.sample_rate/2.0, worker.freq*1e3 + worker.sample_rate/2.0, fft_size) / 1e6
            freq_plot_curve.setData(f, PSD_avg)
            freq_plot.setXRange(worker.freq*1e3/1e6 - worker.sample_rate/2e6, worker.freq*1e3/1e6 + worker.sample_rate/2e6)

        def waterfall_plot_callback(spectrogram):
            imageitem.setImage(spectrogram, autoLevels=False)
            sigma = np.std(spectrogram)
            mean = np.mean(spectrogram)
            self.spectrogram_min = mean - 2*sigma
            self.spectrogram_max = mean + 2*sigma

        def end_of_run_callback():
            QTimer.singleShot(0, worker.run)

        worker.time_plot_update.connect(time_plot_callback)
        worker.freq_plot_update.connect(freq_plot_callback)
        worker.waterfall_plot_update.connect(waterfall_plot_callback)
        worker.end_of_run.connect(end_of_run_callback)

        self.sdr_thread.started.connect(worker.run)
        self.sdr_thread.start()


app = QApplication([])
window = MainWindow()
window.show()
signal.signal(signal.SIGINT, signal.SIG_DFL)
app.exec()

if sdr_type == "usrp":
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)