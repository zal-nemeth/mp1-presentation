
# === 1. Load data ===
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# === 1. Load data ===
# Transmission Spectrum without Circulators
# file_path = '/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/mp1-presentation/measurements/wavelength_sweep/2025-05-01-11-09-39mpw2_no600_f1_0_t1_0_wgw_1'
# Pump Transmission Spectrum with Circulators
file_path = '/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/mp1-presentation/measurements/wavelength_sweep/2025-05-16-18-43-58mpw2_no600_f1_0_t1_0_wgw_1'
file_path = "/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/mp1-presentation/measurements/wavelength_sweep/2025-05-20-18-34-22mpw2_no600_f1_0_t1_0_wgw_1"

with open(file_path, 'r') as f:
    lam_line = f.readline().strip()   # λ in nm
    t_line   = f.readline().strip()   # T in dB

λ = np.fromstring(lam_line, sep=' ')
T = np.fromstring(t_line, sep=' ')

# === 2. Find dips deeper than –42 dB ===
invT = -T
peaks, props = find_peaks(invT, height=26, prominence=3)

# === 3. Compute FWHM and Q for each dip ===
# Use prominence-based height for width calculation
widths, width_heights, left_ips, right_ips = peak_widths(
    invT,
    peaks,
    rel_height=0.5,
    wlen=1000,
    # prominence_data=(props['prominences'], props['left_bases'], props['right_bases'])
)
λ_left  = np.interp(left_ips,  np.arange(len(λ)), λ)
λ_right = np.interp(right_ips, np.arange(len(λ)), λ)
fwhm_nm = λ_right - λ_left
λ0      = λ[peaks]
Q       = λ0 / fwhm_nm

# === 4. Compute FSR ===
order      = np.argsort(λ0)
λ0_sorted  = λ0[order]
fwhm_sorted= fwhm_nm[order]
Q_sorted   = Q[order]
FSR        = np.diff(λ0_sorted)

# === 5. Print results with nm units ===
print(f"{'λ₀ (nm)':>12}  {'FWHM (nm)':>12}  {'Q-factor':>10}")
for lam_cen, w, q in zip(λ0_sorted, fwhm_sorted, Q_sorted):
    print(f"{lam_cen*1e9:12.4f}  {w*1e9:12.6f}  {q:10.2f}")

print("\nFSR between successive dips (nm):")
for d in FSR:
    print(f"{d*1e9:12.6f}")

# === 6. Plot and annotate ===
plt.figure(figsize=(14,10))
plt.plot(λ, T, '-', label='Transmission')
plt.plot(λ0, T[peaks], 'ro', label='Resonance dips')
for i in range(len(peaks)):
    plt.hlines(
        y=-width_heights[i],
        xmin=λ_left[i], xmax=λ_right[i],
        color='r', linewidth=1.5
    )
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission (dB)')
plt.title('Resonance Dips & FWHM')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks, savgol_filter

# # === 1. Load data ===
# file_path = '/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/mp1-presentation/measurements/2025-05-16-18-43-58mpw2_no600_f1_0_t1_0_wgw_1'
# with open(file_path, 'r') as f:
#     lam_line = f.readline().strip()   # λ in nm
#     t_line   = f.readline().strip()   # T in dB

# λ = np.fromstring(lam_line, sep=' ')
# T = np.fromstring(t_line, sep=' ')

# # Apply minimal smoothing to reduce noise but preserve peak shapes
# T_smooth = savgol_filter(T, window_length=11, polyorder=2)

# # Calculate derivative to find regions of steep change
# dT_dλ = np.gradient(T_smooth, λ)

# # Find dips deeper than specified threshold
# invT = -T_smooth
# peaks, props = find_peaks(invT, height=39, prominence=3)

# # Calculate 3dB widths based on derivative information
# refined_peaks = []
# refined_left_ips = []
# refined_right_ips = []
# refined_width_heights = []

# for peak_idx in peaks:
#     # Define the region around the peak
#     window_size = 500
#     left_bound = max(0, peak_idx - window_size)
#     right_bound = min(len(λ) - 1, peak_idx + window_size)
    
#     # Use derivative to find the steep edges
#     derivative_threshold = np.std(dT_dλ) * 1
    
#     # Find left edge (steepest negative slope)
#     left_candidates = np.where(dT_dλ[left_bound:peak_idx] < -derivative_threshold)[0]
#     left_edge = left_bound + left_candidates[0] if len(left_candidates) > 0 else left_bound
    
#     # Find right edge (steepest positive slope)
#     right_candidates = np.where(dT_dλ[peak_idx:right_bound] > derivative_threshold)[0]
#     right_edge = peak_idx + right_candidates[0] if len(right_candidates) > 0 else right_bound
    
#     # Calculate 3dB points from the minimum of the dip
#     T_min = T_smooth[peak_idx]
#     T_3dB = T_min + 3.0
    
#     # Find left 3dB point within the steep region
#     for i in range(peak_idx, left_edge, -1):
#         if T_smooth[i] >= T_3dB:
#             left_3dB = i
#             break
#     else:
#         left_3dB = left_edge
    
#     # Find right 3dB point within the steep region
#     for i in range(peak_idx, right_edge):
#         if T_smooth[i] >= T_3dB:
#             right_3dB = i
#             break
#     else:
#         right_3dB = right_edge
    
#     refined_peaks.append(peak_idx)
#     refined_left_ips.append(left_3dB)
#     refined_right_ips.append(right_3dB)
#     refined_width_heights.append(-T_3dB)

# # Convert to numpy arrays
# refined_peaks = np.array(refined_peaks)
# refined_left_ips = np.array(refined_left_ips)
# refined_right_ips = np.array(refined_right_ips)
# refined_width_heights = np.array(refined_width_heights)

# # Calculate FWHM and Q factors
# λ_left = np.interp(refined_left_ips, np.arange(len(λ)), λ)
# λ_right = np.interp(refined_right_ips, np.arange(len(λ)), λ)
# fwhm_nm = λ_right - λ_left
# λ0 = λ[refined_peaks]
# Q = λ0 / fwhm_nm

# # Compute FSR
# order = np.argsort(λ0)
# λ0_sorted = λ0[order]
# fwhm_sorted = fwhm_nm[order]
# Q_sorted = Q[order]
# FSR = np.diff(λ0_sorted)

# # Print results with nm units
# print(f"{'λ₀ (nm)':>12}  {'FWHM (nm)':>12}  {'Q-factor':>10}")
# for lam_cen, w, q in zip(λ0_sorted, fwhm_sorted, Q_sorted):
#     print(f"{lam_cen*1e9:12.4f}  {w*1e9:12.6f}  {q:10.2f}")

# print("\nFSR between successive dips (nm):")
# for d in FSR:
#     print(f"{d*1e9:12.6f}")

# # === 6. Plot and annotate ===
# plt.figure(figsize=(14,10))
# plt.plot(λ, T, '-', label='Transmission')
# plt.plot(λ0, T[peaks], 'ro', label='Resonance dips')
# for i in range(len(peaks)):
#     plt.hlines(
#         y=-refined_width_heights[i],
#         xmin=λ_left[i], xmax=λ_right[i],
#         color='r', linewidth=1.5
#     )
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Transmission (dB)')
# plt.title('Resonance Dips & FWHM')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# # plt.show()


# # === 1. Load data ===
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks, peak_widths

# # Transmission Spectrum file
# file_path = '/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/mp1-presentation/measurements/2025-05-16-18-43-58mpw2_no600_f1_0_t1_0_wgw_1'
# with open(file_path, 'r') as f:
#     lam_line = f.readline().strip()   # λ in nm
#     t_line   = f.readline().strip()   # T in dB

# λ = np.fromstring(lam_line, sep=' ')
# T = np.fromstring(t_line, sep=' ')

# # === 2. Find dips deeper than –42 dB ===
# invT = -T
# peaks, props = find_peaks(invT, height=54, prominence=3)

# # === 3. Compute FWHM and Q for each dip ===
# widths, width_heights, left_ips, right_ips = peak_widths(
#     invT,
#     peaks,
#     rel_height=0.5,
#     prominence_data=(props['prominences'], props['left_bases'], props['right_bases'])
# )
# λ_left  = np.interp(left_ips,  np.arange(len(λ)), λ)
# λ_right = np.interp(right_ips, np.arange(len(λ)), λ)
# fwhm_nm = λ_right - λ_left
# λ0      = λ[peaks]
# Q       = λ0 / fwhm_nm

# # === 4. Compute FSR ===
# order      = np.argsort(λ0)
# λ0_sorted  = λ0[order]
# fwhm_sorted= fwhm_nm[order]
# Q_sorted   = Q[order]
# FSR        = np.diff(λ0_sorted)

# # === 5. Print results with nm units ===
# print(f"{'λ₀ (nm)':>12}  {'FWHM (nm)':>12}  {'Q-factor':>10}")
# for lam_cen, w, q in zip(λ0_sorted, fwhm_sorted, Q_sorted):
#     print(f"{lam_cen*1e9:12.4f}  {w*1e9:12.6f}  {q:10.2f}")

# print("\nFSR between successive dips (nm):")
# for d in FSR:
#     print(f"{d*1e9:12.6f}")

# # === 6. Plot and annotate ===
# plt.figure(figsize=(14, 10))
# plt.plot(λ, T, '-', label='Transmission')

# # === 6. Plot and annotate ===
# plt.figure(figsize=(14, 10))
# plt.plot(λ, T, '-', label='Transmission')

# # 1) centre of each dip
# plt.plot(λ0, T[peaks], 'ro', label='Resonance centre')

# # 2) FWHM horizontal lines
# for i in range(len(peaks)):
#     plt.hlines(
#         y=-width_heights[i],
#         xmin=λ_left[i], xmax=λ_right[i],
#         color='r', linewidth=1.5,
#         label='FWHM' if i == 0 else None
#     )

# # 3) base points (bottoms of invT peaks = left/right bases of each dip)
# left_bases  = props['left_bases']
# right_bases = props['right_bases']

# plt.plot(λ[left_bases],  T[left_bases],  'gv', markersize=8, label='Left base')
# plt.plot(λ[right_bases], T[right_bases], 'g^', markersize=8, label='Right base')

# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Transmission (dB)')
# plt.title('Resonance Dips with FWHM and Base Points')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks, peak_prominences, peak_widths

# # === 1. Load data ===
# # file_path = '/home/…/2025-05-16-18-43-58mpw2_no600_f1_0_t1_0_wgw_1'
# with open(file_path, 'r') as f:
#     lam_line = f.readline().strip()
#     t_line   = f.readline().strip()

# λ = np.fromstring(lam_line, sep=' ')
# T = np.fromstring(t_line,   sep=' ')
# invT = -T

# # === 2. Detect peaks with a minimum prominence of 3 dB ===
# peaks, peak_props = find_peaks(invT, height=54, prominence=3)

# # === 3. Recompute exact prominences & their bases ===
# prominences, left_bases, right_bases = peak_prominences(invT, peaks, wlen=1000)

# # === 4. Build an array of absolute heights at which we want the width ===
# #    (i.e. base_height + actual prominence for each peak)
# base_heights = invT[left_bases]
# height_at_prom = base_heights + prominences

# # === 5. Measure widths at those exact heights ===
# #    Pass the prominence_data tuple back in so peak_widths can compute
# widths, width_heights, left_ips, right_ips = peak_widths(
#     invT,
#     peaks,
#     prominence_data=(prominences, left_bases, right_bases)
# )

# # Convert index‐positions to actual wavelength (nm)
# λ_left  = np.interp(left_ips,  np.arange(len(λ)), λ)
# λ_right = np.interp(right_ips, np.arange(len(λ)), λ)
# fwhm_nm = λ_right - λ_left
# λ0      = λ[peaks]

# # === 6. Plot everything ===
# plt.figure(figsize=(12, 6))
# plt.plot(λ, T, '-', label='Transmission (dB)')

# # Mark the resonance centres
# plt.plot(λ0, T[peaks], 'ro', label='Resonance centre')

# # Mark the bases used for prominence
# plt.plot(λ[left_bases],  T[left_bases],  'gv', label='Left base')
# plt.plot(λ[right_bases], T[right_bases], 'g^', label='Right base')

# # Draw the “width at prominence” lines
# for i in range(len(peaks)):
#     plt.hlines(
#         y=-height_at_prom[i], 
#         xmin=λ_left[i], xmax=λ_right[i], 
#         color='r', linewidth=1.5,
#         label='Width at prom.' if i == 0 else None
#     )

# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Transmission (dB)')
# plt.title('Resonance Dips: widths measured at exact prominence')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
