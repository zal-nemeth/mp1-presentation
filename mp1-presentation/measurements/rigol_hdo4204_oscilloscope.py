# import pyvisa
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import os
# import csv
# import time

# # === USER CONFIGURABLE ===
# resource = "TCPIP0::169.254.112.67::inst0::INSTR"
# channel_to_fft = 2             # Channel to run FFT on
# total_duration = 20.0          # Total time to record (s)
# sleep_between_reads = 0.00001  # Delay between reads (s)
# top_n_peaks_to_print = 3       # Number of FFT peaks to print
# min_peak_freq = 1000             # Min freq to consider (Hz)
# max_peak_freq = 10000          # Max freq to consider (Hz)
# output_dir = "waveform_data"
# # ==========================

# # Connect to oscilloscope
# rm = pyvisa.ResourceManager()
# scope = rm.open_resource(resource)
# scope.timeout = 5000
# scope.encoding = 'ascii'
# print("Connected to:", scope.query("*IDN?").strip())

# # === UTILITY FUNCTIONS ===
# def get_active_channels(max_channels=4):
#     """Return list of displayed/active channels."""
#     active = []
#     for ch in range(1, max_channels + 1):
#         try:
#             if scope.query(f":CHAN{ch}:DISP?").strip() == '1':
#                 active.append(ch)
#         except Exception as e:
#             print(f"Error checking CH{ch}: {e}")
#     return active

# def get_waveform_data(channel):
#     """Fetch waveform data from a channel."""
#     scope.write(f":WAV:SOUR CHAN{channel}")
#     scope.write(":WAV:MODE NORM")
#     scope.write(":WAV:FORM ASC")
#     data_str = scope.query(":WAV:DATA?")
#     data_str = data_str.strip().split(',')
#     y_data = np.array(data_str, dtype=float)
#     x_increment = float(scope.query(":WAV:XINC?"))
#     x_origin = float(scope.query(":WAV:XOR?"))
#     x_data = x_origin + np.arange(len(y_data)) * x_increment
#     return x_data, y_data, x_increment

# def calculate_fft(y_data, sampling_rate):
#     N = len(y_data)
#     fft_result = np.fft.fft(y_data)
#     fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)
#     return fft_freq[:N // 2], np.abs(fft_result[:N // 2])

# def find_top_peaks(freqs, magnitudes, n=5, min_freq=80, max_freq=10000):
#     valid = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
#     filtered_freqs = freqs[valid]
#     filtered_mags = magnitudes[valid]
#     sorted_idx = np.argsort(filtered_mags)[::-1]
#     return [(filtered_freqs[i], filtered_mags[i]) for i in sorted_idx[:n]]

# # === ACQUISITION ===
# active_channels = get_active_channels()
# print(f"Active channels: {active_channels}")
# waveform_data = {}
# sample_interval = None

# print(f"Starting acquisition for {total_duration}s...")
# start_time = time.time()
# while time.time() - start_time < total_duration:
#     for ch in active_channels:
#         try:
#             _, y_chunk, xinc = get_waveform_data(ch)
#             waveform_data.setdefault(ch, []).extend(y_chunk)
#             if sample_interval is None:
#                 sample_interval = xinc
#         except Exception as e:
#             print(f"Read error on CH{ch}: {e}")
#     time.sleep(sleep_between_reads)

# scope.close()
# print("Acquisition complete.")

# # === SAVE ALL CHANNELS TO CSV ===
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# filename = f"msp_waveforms_{timestamp}.csv"
# filepath = os.path.join(output_dir, filename)
# os.makedirs(output_dir, exist_ok=True)

# try:
#     min_len = min(len(waveform_data[ch]) for ch in active_channels)
#     t_common = np.arange(min_len) * sample_interval
#     with open(filepath, 'w', newline='') as f:
#         writer = csv.writer(f)
#         header = ["Time (s)"] + [f"CH{ch} (V)" for ch in active_channels]
#         writer.writerow(header)
#         for i in range(min_len):
#             row = [t_common[i]] + [waveform_data[ch][i] for ch in active_channels]
#             writer.writerow(row)
#     print(f"Saved waveform data to {filepath}")
# except Exception as e:
#     print(f"Failed to save waveform CSV: {e}")

# # === FFT on SELECTED CHANNEL ===
# y_full = np.array(waveform_data.get(channel_to_fft, []))
# t_full = np.arange(len(y_full)) * sample_interval
# sampling_rate = 1 / sample_interval
# freqs, magnitude = calculate_fft(y_full, sampling_rate)

# top_peaks = find_top_peaks(freqs, magnitude, n=top_n_peaks_to_print,
#                            min_freq=min_peak_freq, max_freq=max_peak_freq)

# print(f"\nTop {top_n_peaks_to_print} frequency peaks (Hz | Magnitude dB):")
# for freq, mag in top_peaks:
#     print(f"{freq:.2f} Hz\t{20 * np.log10(mag + 1e-12):.2f} dB")

# # === TIME-DOMAIN PLOT FOR SELECTED CHANNEL ===
# plt.figure(figsize=(12, 5))
# plt.plot(t_full, y_full, label=f"Channel {channel_to_fft}")
# plt.title(f"Time-Domain Signal - Channel {channel_to_fft}")
# plt.xlabel("Time (s)")
# plt.ylabel("Voltage (V)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # === FFT PLOT ===
# plt.figure(figsize=(12, 10))
# plt.plot(freqs, 20 * np.log10(magnitude + 1e-12), label="FFT Magnitude")
# peak_freqs = [f for f, _ in top_peaks]
# peak_mags_db = [20 * np.log10(m + 1e-12) for _, m in top_peaks]
# plt.scatter(peak_freqs, peak_mags_db, color='red', label="Top Peaks", zorder=5)

# plt.title(f"FFT of Channel {channel_to_fft}")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (dB)")
# plt.xlim(0, max_peak_freq)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

import csv
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyvisa

# === USER CONFIGURABLE ===
oscilloscope_resource = "TCPIP0::169.254.112.67::inst0::INSTR"
laser_resource = "TCPIP0::169.254.246.175::inst0::INSTR"
channel_to_fft = 2
recording_duration = 5.0
sleep_between_reads = 3
top_n_peaks_to_print = 1
min_peak_freq = 5000
max_peak_freq = 7000
wavelengths = [
    None,
    1552.4385,
    None,
    1551,
    1556,
]  # None = Laser OFF, 1550 = Pump, 1551 = Pump + offset
output_dir = "waveform_data"
mod = "mod_150"
# ==========================

# === CONNECT TO INSTRUMENTS ===
rm = pyvisa.ResourceManager()
scope = rm.open_resource(oscilloscope_resource)
laser = rm.open_resource(laser_resource)
scope.timeout = 5000
laser.timeout = 5000
scope.encoding = "ascii"
laser.encoding = "ascii"

print("Connected to:", scope.query("*IDN?").strip())
print("Connected to Laser:", laser.query("*IDN?").strip())


def get_active_channels(max_channels=4):
    active = []
    for ch in range(1, max_channels + 1):
        try:
            if scope.query(f":CHAN{ch}:DISP?").strip() == "1":
                active.append(ch)
        except Exception as e:
            print(f"Error checking channel {ch}: {e}")
    return active


def get_waveform_data(channel):
    scope.write(f":WAV:SOUR CHAN{channel}")
    scope.write(":WAV:MODE NORM")
    scope.write(":WAV:FORM ASC")
    data_str = scope.query(":WAV:DATA?")
    y_data = np.array(data_str.strip().split(","), dtype=float)
    x_increment = float(scope.query(":WAV:XINC?"))
    x_origin = float(scope.query(":WAV:XOR?"))
    x_data = x_origin + np.arange(len(y_data)) * x_increment
    return x_data, y_data, x_increment


def calculate_fft(y_data, sampling_rate):
    N = len(y_data)
    fft_result = np.fft.fft(y_data)
    fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)
    return fft_freq[: N // 2], np.abs(fft_result[: N // 2])


def find_top_peaks(freqs, magnitudes, n=5, min_freq=80, max_freq=10000):
    valid = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    filtered_freqs = freqs[valid]
    filtered_mags = magnitudes[valid]
    sorted_idx = np.argsort(filtered_mags)[::-1]
    return [(filtered_freqs[i], filtered_mags[i]) for i in sorted_idx[:n]]


def set_laser_wavelength(wavelength_nm):
    if wavelength_nm is None:
        laser.write("OUTP0:STAT 0")
    else:
        laser.write(f":SOUR0:WAV {wavelength_nm}NM")
        laser.write("OUTP0:STAT 1")


# === ACQUISITION LOOP FOR EACH LASER STATE ===
all_traces = []
active_channels = get_active_channels()
sample_interval = None
waveform_all_conditions = {}

for i, wl in enumerate(wavelengths):
    label = "no_pump" if wl is None else f"pump_{wl}nm"
    print(f"\n--- Acquiring: {label} ---")
    set_laser_wavelength(wl)
    print("waiting for laser to settle")
    time.sleep(5)  # Give laser some time to stabilize
    # print("waiting for laser to settle")

    waveform_data = {}
    start_time = time.time()
    print("start acquisition")
    while time.time() - start_time < recording_duration:
        for ch in active_channels:
            _, y_chunk, xinc = get_waveform_data(ch)
            if sample_interval is None:
                sample_interval = xinc
            waveform_data.setdefault(ch, []).extend(y_chunk)

    all_traces.append((label, np.array(waveform_data[channel_to_fft][:])))

    # === Save all channel waveforms for this condition ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ch_filename = f"msp_waveforms_{label}_{timestamp}_{mod}.csv"
    ch_filepath = os.path.join(output_dir, ch_filename)
    os.makedirs(output_dir, exist_ok=True)

    try:
        min_len = min(len(waveform_data[ch]) for ch in active_channels)
        t_common = np.arange(min_len) * sample_interval
        with open(ch_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Time (s)"] + [f"CH{ch} (V)" for ch in active_channels]
            writer.writerow(header)
            for i in range(min_len):
                row = [t_common[i]] + [waveform_data[ch][i] for ch in active_channels]
                writer.writerow(row)
        print(f"Saved waveform data to {ch_filepath}")
    except Exception as e:
        print(f"Failed to save waveform CSV for {label}: {e}")

    waveform_all_conditions[label] = waveform_data


# Shutdown
laser.write("OUTP0:STAT 0")
scope.close()
laser.close()

# === SAVE SELECTED CHANNEL FFT COMPARISON CSV ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fft_filename = f"fft_comparison_channel{channel_to_fft}_{timestamp}_{mod}.csv"
fft_filepath = os.path.join(output_dir, fft_filename)

min_len = min(len(y) for _, y in all_traces)
t_common = np.arange(min_len) * sample_interval

try:
    with open(fft_filepath, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Time (s)"] + [label for label, _ in all_traces]
        writer.writerow(header)
        for i in range(min_len):
            row = [t_common[i]] + [y[i] for _, y in all_traces]
            writer.writerow(row)
    print(f"Saved FFT comparison data to {fft_filepath}")
except Exception as e:
    print(f"Failed to save FFT CSV: {e}")

# === FFT and Plotting ===
plt.figure(figsize=(12, 8))
for label, y in all_traces:
    y = y[:min_len]
    freqs, magnitude = calculate_fft(y, 1 / sample_interval)
    top_peaks = find_top_peaks(
        freqs,
        magnitude,
        n=top_n_peaks_to_print,
        min_freq=min_peak_freq,
        max_freq=max_peak_freq,
    )
    peak_freqs = [f for f, _ in top_peaks]
    peak_mags_db = [20 * np.log10(m + 1e-12) for _, m in top_peaks]

    print(f"\nTop {top_n_peaks_to_print} peaks for {label}:")
    for freq, mag in top_peaks:
        print(f"{freq:.2f} Hz\t{20 * np.log10(mag + 1e-12):.2f} dB")

    plt.plot(freqs, 20 * np.log10(magnitude + 1e-12), label=label)
    plt.scatter(peak_freqs, peak_mags_db, color="red", zorder=5)

plt.title("FFT Comparison Across Laser Wavelengths")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, 10000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
