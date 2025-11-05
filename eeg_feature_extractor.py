# CODE FOR FEATURE EXTRACTION FROM EEG

import os
import mne
import numpy as np
from scipy.signal import welch
from tqdm import tqdm
import csv

# Base directory containing subject folders
base_dir = r"D:\SEMESTER PROJECT\Datasets\Alzheimer's disease, Frontotemporal dementia and Healthy subjects\ds004504\derivatives"

# List all subjects (sub-001, sub-002, ...)
subjects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Define EEG frequency bands
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

def bandpower(data, sf, band):
    """Compute mean power of a frequency band using Welch PSD."""
    fmin, fmax = band
    freqs, psd = welch(data, sf, nperseg=sf*2)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.mean(psd[idx])

def process_subject(eeg_file):
    """Load EEG, filter, average reference, compute bandpowers per channel."""
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    raw.filter(0.5, 45.0)  # Bandpass filter
    raw.set_eeg_reference('average')  # Average reference
    data, times = raw.get_data(return_times=True)
    sfreq = raw.info['sfreq']

    features = {}
    for i, ch in enumerate(raw.ch_names):
        features[ch] = {band_name: bandpower(data[i], sfreq, band_range)
                        for band_name, band_range in bands.items()}
    return features

# Prepare to save features
all_features = []
error_log = os.path.join(base_dir, "processing_errors.log")

# Open log file in UTF-8 to avoid Unicode errors
with open(error_log, "w", encoding="utf-8") as f_log:
    for sub in tqdm(subjects, desc="Processing Subjects"):
        eeg_file = os.path.join(base_dir, sub, "eeg", f"{sub}_task-eyesclosed_eeg.set")
        if not os.path.exists(eeg_file):
            f_log.write(f"{sub} -> File missing\n")
            continue
        try:
            features = process_subject(eeg_file)
            # Flatten for CSV: one row per subject
            flat_features = {"Subject": sub}
            for ch, band_dict in features.items():
                for band_name, value in band_dict.items():
                    flat_features[f"{ch}_{band_name}"] = value
            all_features.append(flat_features)
        except Exception as e:
            f_log.write(f"{sub} -> Failed: {e}\n")

# Save all features to CSV
if all_features:
    csv_file = os.path.join(base_dir, "features.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_features[0].keys())
        writer.writeheader()
        writer.writerows(all_features)

print("Feature extraction complete ✅")
print(f"Saved to: {csv_file}")
print(f"Errors logged to: {error_log}")

      