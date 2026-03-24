"""
Configuration: all physical constants, MCS table, turbulence regimes, hyperparameters.
Values from Dindar et al. paper unless noted otherwise.
"""
import numpy as np

# --- FSO Link Parameters (from paper) ---
WAVELENGTH = 1550e-9        # m
LINK_DISTANCE = 800.0       # m
TX_POWER_W = 8e-3           # 8 mW
TX_POWER_DBM = 9.0          # dBm
RX_APERTURE_D = 75e-3       # m (plano-convex lens diameter)
PEAK_RX_POWER_DBM = -3.0    # dBm
LINK_FAILURE_DBM = -27.0    # dBm
FADE_MARGIN_DB = 24.0       # dB
SAMPLE_RATE = 10_000        # Hz (10 kHz photodetector DC monitor)
DT = 1.0 / SAMPLE_RATE     # 0.1 ms

# Empirical SNR model: SNR_dB = 1.04 * P_dBm + 33.1
SNR_SLOPE = 1.04
SNR_INTERCEPT = 33.1

# --- Turbulence outer scale ---
L0 = 25.0  # m (outer scale, Von Kármán)

# --- MCS Table (Table I of paper + outage class) ---
# Class 0 = outage (SNR < 7 dB)
# Classes 1-14 = paper's MCS 0-13
MCS_TABLE = [
    {"class": 0,  "name": "Outage",         "mod": "None",   "rate": 0,    "snr_min": -np.inf, "bits_per_sym": 0, "code_rate": 0},
    {"class": 1,  "name": "QPSK 1/2",       "mod": "QPSK",   "rate": 0.5,  "snr_min": 7.0,     "bits_per_sym": 2, "code_rate": 0.5},
    {"class": 2,  "name": "QPSK 2/3",       "mod": "QPSK",   "rate": 2/3,  "snr_min": 8.0,     "bits_per_sym": 2, "code_rate": 2/3},
    {"class": 3,  "name": "QPSK 3/4",       "mod": "QPSK",   "rate": 0.75, "snr_min": 8.5,     "bits_per_sym": 2, "code_rate": 0.75},
    {"class": 4,  "name": "QPSK 5/6",       "mod": "QPSK",   "rate": 5/6,  "snr_min": 9.5,     "bits_per_sym": 2, "code_rate": 5/6},
    {"class": 5,  "name": "16QAM 1/2",      "mod": "16QAM",  "rate": 0.5,  "snr_min": 13.5,    "bits_per_sym": 4, "code_rate": 0.5},
    {"class": 6,  "name": "16QAM 2/3",      "mod": "16QAM",  "rate": 2/3,  "snr_min": 14.5,    "bits_per_sym": 4, "code_rate": 2/3},
    {"class": 7,  "name": "16QAM 3/4",      "mod": "16QAM",  "rate": 0.75, "snr_min": 15.5,    "bits_per_sym": 4, "code_rate": 0.75},
    {"class": 8,  "name": "16QAM 5/6",      "mod": "16QAM",  "rate": 5/6,  "snr_min": 16.0,    "bits_per_sym": 4, "code_rate": 5/6},
    {"class": 9,  "name": "64QAM 2/3",      "mod": "64QAM",  "rate": 2/3,  "snr_min": 20.0,    "bits_per_sym": 6, "code_rate": 2/3},
    {"class": 10, "name": "64QAM 3/4",      "mod": "64QAM",  "rate": 0.75, "snr_min": 21.0,    "bits_per_sym": 6, "code_rate": 0.75},
    {"class": 11, "name": "64QAM 5/6",      "mod": "64QAM",  "rate": 5/6,  "snr_min": 21.5,    "bits_per_sym": 6, "code_rate": 5/6},
    {"class": 12, "name": "256QAM 2/3",     "mod": "256QAM", "rate": 2/3,  "snr_min": 27.0,    "bits_per_sym": 8, "code_rate": 2/3},
    {"class": 13, "name": "256QAM 3/4",     "mod": "256QAM", "rate": 0.75, "snr_min": 28.0,    "bits_per_sym": 8, "code_rate": 0.75},
    {"class": 14, "name": "256QAM 5/6",     "mod": "256QAM", "rate": 5/6,  "snr_min": 29.0,    "bits_per_sym": 8, "code_rate": 5/6},
]
NUM_CLASSES = len(MCS_TABLE)  # 15
HYSTERESIS_DB = 0.5

# Throughput per MCS class (bits_per_sym * code_rate), normalised to max
MCS_THROUGHPUT = np.array([m["bits_per_sym"] * m["code_rate"] for m in MCS_TABLE])
MAX_THROUGHPUT = MCS_THROUGHPUT.max()

# SNR thresholds array for quick lookup
MCS_SNR_THRESHOLDS = np.array([m["snr_min"] for m in MCS_TABLE])

# --- Turbulence Regimes ---
TURBULENCE_REGIMES = {
    "weak":        {"Cn2": 5e-16, "wind_mean": 2.0,  "wind_std": 0.5},
    "moderate":    {"Cn2": 5e-15, "wind_mean": 5.0,  "wind_std": 1.5},
    "strong":      {"Cn2": 5e-14, "wind_mean": 8.0,  "wind_std": 2.0},
    "very_strong": {"Cn2": 5e-13, "wind_mean": 10.0, "wind_std": 3.0},
}

# --- Dataset / Windowing ---
WINDOW_SIZE = 100       # samples (10 ms)
WINDOW_STRIDE = 50      # samples (5 ms)
SI_WINDOW = 500         # samples for scintillation index (backward-looking)
N_FEATURES = 5          # raw_power_dBm, raw_snr_dB, norm_power, log_SI, power_derivative

# --- Training Hyperparameters ---
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5
LR_PATIENCE = 7
LR_FACTOR = 0.5
GRAD_CLIP = 1.0
DROPOUT = 0.3

# --- Quick mode defaults ---
QUICK_DURATION_S = 30       # seconds per realisation
QUICK_REALISATIONS = 1      # per regime
FULL_DURATION_S = 300       # seconds per realisation  
FULL_REALISATIONS = 3       # per regime
