
import pyvisa
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv
import time

from scipy.signal import butter, filtfilt

from moku.instruments import WaveformGenerator

# === PROXY SETTINGS FOR INSTRUMENT CONTROL ===
# Save original NO_PROXY setting
original_no_proxy = os.environ.get('NO_PROXY', '')

# Temporarily disable proxy for local IPs
# os.environ['NO_PROXY'] = 'localhost,127.0.0.1,169.254.112.97/16'

# === USER CONFIGURABLE ===
oscilloscope_resource = "TCPIP0::169.254.112.67::inst0::INSTR"
laser_resource = "TCPIP0::169.254.246.175::inst0::INSTR"
moku_ip = "169.254.112.97"
channel_to_fft = 2
recording_duration = 20.0
num_averages = 5  # Number of waveform acquisitions to average
sleep_between_reads = 3
top_n_peaks_to_print = 1
# Fast modulation for Kerr
min_peak_freq = 4000
max_peak_freq = 6000
wavelengths = [None, 1552.4425, None, 1552.2, 1552.6]
wavelengths = [None, 1557.4654, None, 1557.3, 1557.6]
output_dir = "waveform_data"
# mod = "smod_4k_pmod_off"
mod = "smod_1K_pmod_600"

# === OPERATION MODE ===
# Options: "read_only" or "full_acquisition"
acquisition_mode = "full_acquisition"
# acquisition_mode = "read_only"
# acquisition_mode = "modulation_response"
# ==========================

# === CONNECT TO INSTRUMENTS ===
rm = pyvisa.ResourceManager()
scope = rm.open_resource(oscilloscope_resource)
laser = rm.open_resource(laser_resource)
scope.timeout = 5000
laser.timeout = 5000
scope.encoding = 'ascii'
laser.encoding = 'ascii'

print("Connected to:", scope.query("*IDN?").strip())
print("Connected to Laser:", laser.query("*IDN?").strip())

def get_active_channels(max_channels=4):
    active = []
    for ch in range(1, max_channels + 1):
        try:
            if scope.query(f":CHAN{ch}:DISP?").strip() == '1':
                active.append(ch)
        except Exception as e:
            print(f"Error checking channel {ch}: {e}")
    return active

def get_waveform_data(channel):
    scope.write(f":WAV:SOUR CHAN{channel}")
    scope.write(":WAV:MODE NORM")
    scope.write(":WAV:FORM ASC")
    data_str = scope.query(":WAV:DATA?")
    y_data = np.array(data_str.strip().split(','), dtype=float)
    x_increment = float(scope.query(":WAV:XINC?"))
    x_origin = float(scope.query(":WAV:XOR?"))
    x_data = x_origin + np.arange(len(y_data)) * x_increment
    return x_data, y_data, x_increment

def calculate_fft(y_data, sampling_rate):
    N = len(y_data)
    fft_result = np.fft.fft(y_data)
    fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)
    return fft_freq[:N // 2], np.abs(fft_result[:N // 2])

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

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def calculate_rms(signal):
    return np.sqrt(np.mean(np.square(signal)))


if acquisition_mode == "read_only":
    print("\n--- Simple Read Mode ---")
    sample_interval = None
    waveform_data = {}
    active_channels = get_active_channels()
    start_time = time.time()
    print("Acquiring data...")

    print(f"Averaging {num_averages} acquisitions over {recording_duration} seconds total...")
    acq_duration = recording_duration / num_averages
    for avg_idx in range(num_averages):
        print(f"  Acquisition {avg_idx + 1}/{num_averages} ({acq_duration:.2f}s each)")
        temp_data = {}
        start_time = time.time()
        while time.time() - start_time < acq_duration:
            for ch in active_channels:
                _, y_chunk, xinc = get_waveform_data(ch)
                if sample_interval is None:
                    sample_interval = xinc
                temp_data.setdefault(ch, []).extend(y_chunk)
        # Convert to array
        for ch in temp_data:
            temp_data[ch] = np.array(temp_data[ch])
            waveform_data.setdefault(ch, []).append(temp_data[ch])

   # Average all acquisitions per channel (safely trim to shortest length)
    for ch in waveform_data:
        min_len = min(len(arr) for arr in waveform_data[ch])
        trimmed = np.stack([arr[:min_len] for arr in waveform_data[ch]])
        waveform_data[ch] = np.mean(trimmed, axis=0)

    # ✅ Only run this once — after averaging all channels
    data_len = len(waveform_data[channel_to_fft])
    print(f"The number of samples acquired is: {data_len}")
    t_common = np.arange(min_len) * sample_interval
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ch_filename = f"read_only_waveform_{timestamp}_{mod}.csv"
    ch_filepath = os.path.join(output_dir, ch_filename)
    os.makedirs(output_dir, exist_ok=True)

    if "test" in ch_filename:
        print("File was not saved.")
    else:
        try:
            with open(ch_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Time (s)"] + [f"CH{ch} (V)" for ch in active_channels]
                writer.writerow(header)
                for i in range(min_len):
                    row = [t_common[i]] + [waveform_data[ch][i] for ch in active_channels]
                    writer.writerow(row)
            print(f"Saved read-only waveform data to {ch_filepath}")
        except Exception as e:
            print(f"Failed to save read-only waveform CSV: {e}")

    # === FFT and Plot ===
    y = waveform_data[channel_to_fft][:min_len]
    rms_value = calculate_rms(y)
    print(f"RMS value (read-only, CH{channel_to_fft}): {rms_value:.6f} V")
    freqs, magnitude = calculate_fft(y, 1 / sample_interval)
    # top_peaks = find_top_peaks(freqs, magnitude, n=top_n_peaks_to_print,
    #                         min_freq=min_peak_freq, max_freq=max_peak_freq)

    # print(f"\nTop {top_n_peaks_to_print} peaks (read-only):")
    # for freq, mag in top_peaks:
    #     print(f"{freq:.2f} Hz\t{20 * np.log10(mag + 1e-12):.2f} dB")

    # plt.figure(figsize=(10, 6))
    # plt.plot(freqs, 20 * np.log10(magnitude + 1e-12), label="Read-Only FFT")
    # plt.title("FFT - Read Only Mode")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude (dB)")
    # plt.xlim(min_peak_freq, max_peak_freq)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

elif acquisition_mode == "modulation_response":

    modulation_frequencies = [
        100, 300, 500, 700, 900, 1_000, 1500, 2_000, 2500, 3_000, 3500, 4_000, 5_000, 7_000, 8_000, 9_000, 10_000,
        12_000, 14_000, 16_000, 18_000, 20_000, 25_000, 30_000, 40_000, 50_000, 60_000, 75_000, 100_000, 
        200_000, 300_000, 400_000, 500_000, 750_000, 1_000_000, 2_000_000, 5000_000
    ]

    results = []

    def acquire_vrms(label=""):
        total_data = []
        sample_interval = None
        for i in range(num_averages):
            chunk_data = []
            start_time = time.time()
            while time.time() - start_time < (recording_duration / num_averages):
                _, y_chunk, xinc = get_waveform_data(channel_to_fft)
                chunk_data.extend(y_chunk)
                if sample_interval is None:
                    sample_interval = xinc
            total_data.append(np.array(chunk_data))
        min_len = min(len(x) for x in total_data)
        stacked = np.stack([d[:min_len] for d in total_data])
        mean_data = np.mean(stacked, axis=0)
        return calculate_rms(mean_data)

    print("\n--- MODULATION RESPONSE MODE ---")

    # 1. Laser OFF (baseline)
    print("Turning laser OFF to get baseline power...")
    set_laser_wavelength(None)
    time.sleep(5)
    vrms_baseline = acquire_vrms("baseline")
    print(f"Baseline Vrms: {vrms_baseline:.6f} V")

    # 2. Laser ON at fixed power/wavelength
    print("Turning laser ON at 15.849 mW / 1552.4415 nm...")
    laser.write(":SOUR0:POW 15.849MW")
    set_laser_wavelength(1557.4654)
    time.sleep(5)
    vrms_laser_on = acquire_vrms("laser_on")
    print(f"Laser ON Vrms: {vrms_laser_on:.6f} V")

    # # 3. Setup Moku WaveformGenerator
    # for freq in modulation_frequencies:
    #     print(f"Setting Moku square wave at {freq} Hz...")
    #     os.environ['NO_PROXY'] = 'localhost,127.0.0.1,169.254.112.97/16'
    #     wg = WaveformGenerator(moku_ip, force_connect=True)
    #     try:
    #         wg.generate_waveform(channel=1,
    #                              amplitude=5,
    #                              type='Square',
    #                              frequency=freq,
    #                              duty=50)
    #     finally:
    #         del wg  # ensure disconnect
    #     os.environ['NO_PROXY'] = original_no_proxy
    #     print("Measuring with oscilloscope...")
    #     time.sleep(5)
    #     vrms_modulated = acquire_vrms(f"mod_{freq}")
    #     results.append((freq, vrms_modulated))
    #     print(f"  -> Vrms at {freq} Hz: {vrms_modulated:.6f} V")
    for freq in modulation_frequencies:
        print(f"\nSetting Moku square wave at {freq} Hz...")

        # 1. Disable proxy for Moku
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1,169.254.112.97/16'
        wg = WaveformGenerator(moku_ip, force_connect=True)
        try:
            wg.generate_waveform(channel=1,
                                 amplitude=5,
                                 type='Square',
                                 frequency=freq,
                                 duty=50)
        finally:
            del wg  # force release connection

        # 2. Re-enable proxy
        os.environ['NO_PROXY'] = original_no_proxy

        # 3. Wait and measure
        print("Waiting for modulation to stabilize...")
        time.sleep(3)

        # --- Acquire waveform data from scope ---
        print("Reading waveform from oscilloscope...")
        sample_interval = None
        all_y_data = []

        for i in range(num_averages):
            chunk = []
            start_time = time.time()
            while time.time() - start_time < (recording_duration / num_averages):
                _, y_chunk, xinc = get_waveform_data(channel_to_fft)
                chunk.extend(y_chunk)
                if sample_interval is None:
                    sample_interval = xinc
            all_y_data.append(np.array(chunk))

        min_len = min(len(x) for x in all_y_data)
        avg_y = np.mean([x[:min_len] for x in all_y_data], axis=0)
        vrms_modulated = calculate_rms(avg_y)
        results.append((freq, vrms_modulated))
        print(f"  -> Vrms at {freq} Hz: {vrms_modulated:.6f} V")

        # --- Save waveform to CSV ---
        time_axis = np.arange(min_len) * sample_interval
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        waveform_filename = f"mod_waveform_{mod}_{freq}Hz_{timestamp}.csv"
        waveform_path = os.path.join(output_dir, waveform_filename)

        try:
            with open(waveform_path, 'w', newline='') as wf:
                writer = csv.writer(wf)
                writer.writerow(["Time (s)", "Voltage (V)"])
                writer.writerows(zip(time_axis, avg_y))
            print(f"Saved waveform for {freq} Hz to {waveform_path}")
        except Exception as e:
            print(f"Failed to save waveform for {freq} Hz: {e}")

        

    # Save to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"modulation_response_{timestamp}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency (Hz)", "Vrms (V)"])
        for row in results:
            writer.writerow(row)
    print(f"Saved modulation response to {csv_filepath}")

    # Plot the response
    freqs, vrms_vals = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, vrms_vals, marker='o')
    plt.xscale('log')
    plt.xlabel("Modulation Frequency (Hz)")
    plt.ylabel("Vrms (V)")
    plt.title("Modulation Response")
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

else:
    all_traces = []
    active_channels = get_active_channels()
    sample_interval = None
    waveform_all_conditions = {}

    for i, wl in enumerate(wavelengths):
        label = "no_pump" if wl is None else f"pump_{wl}nm"
        print(f"\n--- Acquiring: {label} ---")
        set_laser_wavelength(wl)
        print("waiting for laser to settle")
        time.sleep(5)

        waveform_data = {}
        num_averages = 5  # Add this near your other config options
        print(f"Averaging {num_averages} acquisitions over {recording_duration} seconds total...")
        acq_duration = recording_duration / num_averages

        for avg_idx in range(num_averages):
            print(f"  Acquisition {avg_idx + 1}/{num_averages} ({acq_duration:.2f}s each)")
            temp_data = {}
            start_time = time.time()
            while time.time() - start_time < acq_duration:
                for ch in active_channels:
                    _, y_chunk, xinc = get_waveform_data(ch)
                    if sample_interval is None:
                        sample_interval = xinc
                    temp_data.setdefault(ch, []).extend(y_chunk)
            for ch in temp_data:
                temp_data[ch] = np.array(temp_data[ch])
                waveform_data.setdefault(ch, []).append(temp_data[ch])

        # Safely average all collected acquisitions per channel
        for ch in waveform_data:
            min_len = min(len(arr) for arr in waveform_data[ch])
            trimmed = np.stack([arr[:min_len] for arr in waveform_data[ch]])
            waveform_data[ch] = np.mean(trimmed, axis=0)
            avg_waveform = waveform_data[channel_to_fft][:min_len]
            rms_value = calculate_rms(avg_waveform)
            print(f"RMS value for {label} (CH{channel_to_fft}): {rms_value:.6f} V")

            # Now append for FFT comparison
            all_traces.append((label, avg_waveform))



        # all_traces.append((label, np.array(waveform_data[channel_to_fft][:])))

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ch_filename = f"msp_waveforms_{label}_{timestamp}_{mod}.csv"
        ch_filepath = os.path.join(output_dir, ch_filename)
        os.makedirs(output_dir, exist_ok=True)

        if "test" in ch_filename:
            print("File was not saved.")
        else:
            try:
                min_len = min(len(waveform_data[ch]) for ch in active_channels)
                t_common = np.arange(min_len) * sample_interval
                with open(ch_filepath, 'w', newline='') as f:
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

    # Save FFT comparison
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fft_filename = f"fft_comparison_channel{channel_to_fft}_{timestamp}_{mod}.csv"
    fft_filepath = os.path.join(output_dir, fft_filename)

    min_len = min(len(y) for _, y in all_traces)
    t_common = np.arange(min_len) * sample_interval


    if "test" in fft_filename:
        print("FFT file was not saved.")
    else:
        try:
            with open(fft_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Time (s)"] + [label for label, _ in all_traces]
                writer.writerow(header)
                for i in range(min_len):
                    row = [t_common[i]] + [y[i] for _, y in all_traces]
                    writer.writerow(row)
            print(f"Saved FFT comparison data to {fft_filepath}")
        except Exception as e:
            print(f"Failed to save FFT CSV: {e}")

    # Plot FFT comparison
    plt.figure(figsize=(12, 8))
    for label, y in all_traces:
        y = y[:min_len]
        freqs, magnitude = calculate_fft(y, 1 / sample_interval)
        top_peaks = find_top_peaks(freqs, magnitude, n=top_n_peaks_to_print,
                                   min_freq=min_peak_freq, max_freq=max_peak_freq)
        peak_freqs = [f for f, _ in top_peaks]
        peak_mags_db = [20 * np.log10(m + 1e-12) for _, m in top_peaks]

        print(f"\nTop {top_n_peaks_to_print} peaks for {label}:")
        for freq, mag in top_peaks:
            print(f"{freq:.2f} Hz\t{20 * np.log10(mag + 1e-12):.2f} dB")

        plt.plot(freqs, 20 * np.log10(magnitude + 1e-12), label=label)
        plt.scatter(peak_freqs, peak_mags_db, color='red', zorder=5)

    plt.title("FFT Comparison Across Laser Wavelengths")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(min_peak_freq, max_peak_freq)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# # Your setup code for the waveform generator here
# wg.generate_waveform(channel=2,
#                      amplitude=5,
#                      type='Square',
#                      frequency=600,
#                      duty=50)
# # wg.
# # wg.(channel=1, enabled=True)
# print("Waveform generator set up successfully.")
