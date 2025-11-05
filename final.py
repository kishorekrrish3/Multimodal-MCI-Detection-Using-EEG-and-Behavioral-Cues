# ==============================================================================
# FINAL PROJECT: MULTIMODAL MCI/DEMENTIA DETECTION
#
# This single-file Streamlit application combines:
# 1. Phase 1: EEG Analysis (from app.py)
# 2. Phase 2: Behavioral Analysis (from phase2_mci_detection.py)
#
# NEW FEATURES ADDED:
# - Automatic audio extraction from video files (using ffmpeg via subprocess)
# - Speech-to-Text (STT) transcription from extracted audio (using speech_recognition)
# - A 3-model architecture (EEG, Behavioral-Visual, Textual)
# - A final weighted late-fusion module that combines all three models,
#   giving higher weight to the EEG results as requested.
# ==============================================================================

import streamlit as st
import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import json
import time
import cv2
import librosa
import soundfile as sf
from datetime import datetime
from scipy.signal import welch
import shutil  # NEW: For checking ffmpeg availability
import subprocess # NEW: For running ffmpeg

# --- ML/AI Libraries ---
import mne
from catboost import CatBoostClassifier

# --- Phase 2 Libraries (with new additions) ---
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- NEW: Audio/Text Processing Libraries ---
# Check for ffmpeg executable
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
    
try:
    import speech_recognition as sr
    SPEECH_REC_AVAILABLE = True
except ImportError:
    SPEECH_REC_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

# --- General Config ---
st.set_page_config(
    page_title="Multimodal MCI Predictor",
    layout="wide"
)

# --- Phase 1: EEG Config ---
EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# --- Phase 2: Behavioral Config ---
DATA_DIR = Path("multimodal_mci_data")
DATA_DIR.mkdir(exist_ok=True)
EEG_MODEL_FILE = "catboost_eeg_model.cbm"
BEHAVIORAL_MODEL_FILE = DATA_DIR / "behavioral_model.joblib"
BEHAVIORAL_SCALER_FILE = DATA_DIR / "behavioral_scaler.joblib"
TEXT_MODEL_FILE = DATA_DIR / "text_model.joblib"
TEXT_SCALER_FILE = DATA_DIR / "text_scaler.joblib"


# --- NEW: Standardized Class Map for Final Fusion ---
# We will map all model outputs to this 3-class system
FINAL_CLASSES = ["Healthy", "MCI", "Dementia"]

# Phase 1 EEG model has a different map, which we'll convert
EEG_DIAGNOSIS_MAP = {0: "Alzheimer's Disease", 1: "Frontotemporal Dementia", 2: "Healthy"}
EEG_COLOR_MAP = {
    "Healthy": "#2ecc71",
    "Alzheimer's Disease": "#e74c3c",
    "Frontotemporal Dementia": "#f1c40f",
    "Unknown": "#95a5a6"
}

# 10 Cognitive Prompts
COGNITIVE_PROMPTS = [
    {"id": 1, "text": "Please state your full name and age.", "domain": "Orientation", "duration": 10},
    {"id": 2, "text": "Repeat after me: Apple, Car, Blue", "domain": "Memory", "duration": 10},
    {"id": 3, "text": "Count backwards from 20 to 15 aloud.", "domain": "Attention", "duration": 10},
    {"id": 4, "text": "If you had 2 apples and I gave you 3 more, how many would you have?", "domain": "Arithmetic", "duration": 10},
    {"id": 5, "text": "What day and month is it today?", "domain": "Orientation", "duration": 10},
    {"id": 6, "text": "Which word is different: dog, cat, chair, bird?", "domain": "Executive", "duration": 10},
    {"id": 7, "text": "Spell the word WORLD backwards.", "domain": "Attention", "duration": 10},
    {"id": 8, "text": "What is 7 plus 8?", "domain": "Arithmetic", "duration": 10},
    {"id": 9, "text": "Name as many animals as you can in 10 seconds.", "domain": "Fluency", "duration": 15},
    {"id": 10, "text": "Tell me what you did this morning in one sentence.", "domain": "Memory", "duration": 15}
]

# ==============================================================================
# HELPER FUNCTIONS (Combined)
# ==============================================================================

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# ==============================================================================
# PHASE 1: EEG FEATURE EXTRACTION FUNCTIONS
# ==============================================================================

def bandpower(data, sf, band):
    """Compute mean power of a frequency band using Welch PSD."""
    fmin, fmax = band
    freqs, psd = welch(data, sf, nperseg=sf*2)
    # Find frequency indices
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    if np.sum(idx_band) == 0:
        return 0.0 # Avoid errors if band is empty
    return np.mean(psd[idx_band])

def extract_eeg_features(eeg_file, progress_callback=None):
    """Load EEG, filter, re-reference, and compute bandpowers per channel."""
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    raw.filter(0.5, 45.0, verbose=False)
    raw.set_eeg_reference('average', verbose=False)
    data = raw.get_data()
    sfreq = raw.info['sfreq']

    features = {}
    for i, ch in enumerate(raw.ch_names):
        band_values = {}
        for band_name, band_range in EEG_BANDS.items():
            val = bandpower(data[i], sfreq, band_range)
            band_values[band_name] = val
        features[ch] = band_values
        if progress_callback:
            progress_callback()

    # Flatten features for wide format (CSV)
    flat_features = {}
    for ch, band_dict in features.items():
        for band_name, value in band_dict.items():
            flat_features[f"{ch}_{band_name}"] = value
    return flat_features

# ==============================================================================
# PHASE 2: BEHAVIORAL FEATURE EXTRACTION FUNCTIONS
# ==============================================================================

def extract_visual_features(video_path):
    """Extract facial and emotion features from video using DeepFace"""
    features = {}
    if not DEEPFACE_AVAILABLE:
        st.warning(f"DeepFace not available. Skipping visual features for {video_path.name}")
        return features
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        emotions_list = []
        ages_list = []
        frame_count = 0
        sample_rate = 10  # Process every 10th frame
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % sample_rate != 0:
                continue
            
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion', 'age'], 
                                              enforce_detection=False, silent=True)
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                emotions_list.append(analysis['emotion'])
                ages_list.append(analysis['age'])
            except Exception:
                continue
        cap.release()
        
        if emotions_list:
            emotion_df = pd.DataFrame(emotions_list).fillna(0)
            for emotion in emotion_df.columns:
                features[f'vis_{emotion.lower()}_mean'] = float(emotion_df[emotion].mean())
                features[f'vis_{emotion.lower()}_std'] = float(emotion_df[emotion].std())
            features['vis_age_mean'] = float(np.mean(ages_list))
            features['vis_age_std'] = float(np.std(ages_list))
    except Exception as e:
        st.error(f"Error extracting visual features: {e}")
    return features


def extract_audio_features(audio_path):
    """Extract audio features using librosa (non-textual)"""
    features = {}
    try:
        y, sr = librosa.load(str(audio_path), sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        features['aud_duration'] = float(duration)
        
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.mean(rms) * 0.3
        voiced_frames = rms > threshold
        features['aud_voice_activity_ratio'] = float(np.sum(voiced_frames) / len(voiced_frames))
        
        # Pause detection
        frame_length = 2048
        hop_length = 512
        pauses = []
        pause_start = None
        for i, silent in enumerate(rms < threshold):
            if silent and pause_start is None:
                pause_start = i
            elif not silent and pause_start is not None:
                pause_length = (i - pause_start) * hop_length / sr
                pauses.append(pause_length)
                pause_start = None
        
        features['aud_pause_count'] = int(len(pauses))
        features['aud_avg_pause_len'] = float(np.mean(pauses)) if pauses else 0.0
        features['aud_max_pause_len'] = float(np.max(pauses)) if pauses else 0.0
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'aud_mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
        
        features['aud_rms_mean'] = float(np.mean(rms))
        features['aud_rms_std'] = float(np.std(rms))
        
    except Exception as e:
        st.error(f"Error extracting audio features: {e}")
    return features

# --- NEW: Text Feature Extraction Function ---
def extract_text_features(prompt_dir, prompt_id):
    """Extract features from transcribed text"""
    features = {}
    text_path = prompt_dir / "transcription.txt"
    
    if not text_path.exists():
        return features
        
    with open(text_path, 'r') as f:
        text = f.read().lower()

    filler_words = ["um", "uh", "ah", "like", "you know", "so", "well"]
    words = text.split()
    word_count = len(words)
    
    features['text_word_count'] = word_count
    features['text_char_count'] = len(text)
    
    filler_count = sum(text.count(word) for word in filler_words)
    features['text_filler_count'] = filler_count
    features['text_filler_ratio'] = (filler_count / word_count) if word_count > 0 else 0
    
    unique_words = len(set(words))
    features['text_ttr'] = (unique_words / word_count) if word_count > 0 else 0 # Type-Token Ratio
    
    # Prompt-specific features
    if prompt_id == 9: # Animal fluency
        features['text_fluency_count'] = unique_words
        
    if prompt_id == 7: # Spell WORLD backwards
        features['text_world_correct'] = 1 if "d" in text and "l" in text and "r" in text and "o" in text and "w" in text else 0

    return features


def compute_temporal_features(response_times):
    """Compute temporal features across all prompts"""
    features = {}
    if len(response_times) > 0:
        features['temp_mean_rt'] = float(np.mean(response_times))
        features['temp_std_rt'] = float(np.std(response_times))
        if len(response_times) > 1:
            x = np.arange(len(response_times))
            slope, _ = np.polyfit(x, response_times, 1)
            features['temp_rt_slope'] = float(slope) # Fatigue indicator
        else:
            features['temp_rt_slope'] = 0.0
    return features


def aggregate_subject_features(subject_dir):
    """Aggregate all features for a subject"""
    all_features = {}
    response_times = []
    
    for prompt_id in range(1, 11):
        prompt_dir = subject_dir / f"prompt_{prompt_id}"
        if not prompt_dir.exists():
            continue
        
        feature_file = prompt_dir / "features.json"
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                prompt_features = json.load(f)
                
                if 'response_time' in prompt_features:
                    response_times.append(prompt_features['response_time'])
                
                for key, value in prompt_features.items():
                    if key != 'response_time':
                        all_features[f'p{prompt_id}_{key}'] = value
    
    temporal_features = compute_temporal_features(response_times)
    all_features.update(temporal_features)
    return all_features


# ==============================================================================
# PHASE 2: CLASSIFICATION FUNCTIONS
# ==============================================================================

def train_rf_model(features_df, labels):
    """Train a Random Forest classifier"""
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn not installed. Cannot train model.")
        return None, None
    
    X = features_df.fillna(0)
    y = labels
    
    # Ensure all feature names are strings (CatBoost/RF compatibility)
    X.columns = [str(c) for c in X.columns]
    
    if len(set(y)) < 2:
        st.error(f"Cannot train model. Only one class ('{set(y).pop()}') present in labels.")
        return None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Store feature names in the model for prediction
    model.feature_names_in_ = features_df.columns.tolist()
    
    return model, scaler


def predict_cognitive_status(features_df, model, scaler):
    """Predict cognitive status from features"""
    if model is None or scaler is None:
        return None, None

    # Ensure features_df has the same columns as the model
    model_features = model.feature_names_in_
    features_df_aligned = pd.DataFrame(columns=model_features)
    features_df_aligned = pd.concat([features_df_aligned, features_df], axis=0)
    features_df_aligned = features_df_aligned[model_features].fillna(0)
    
    X_scaled = scaler.transform(features_df_aligned)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    return prediction, probabilities


# ==============================================================================
# STREAMLIT UI - MAIN FUNCTION
# ==============================================================================

def main():

    st.sidebar.title("🧠 Multimodal Mild Cognitive Impairment Detection using EEG and Behavioral Cues")
    st.sidebar.markdown("---")
    
    # Check for dependencies
    st.sidebar.subheader("System Status")
    if not DEEPFACE_AVAILABLE:
        st.sidebar.warning("DeepFace not found. Visual analysis disabled.")
    if not SKLEARN_AVAILABLE:
        st.sidebar.warning("scikit-learn not found. Model training/prediction disabled.")
    if not FFMPEG_AVAILABLE:
        st.sidebar.warning("ffmpeg not found in PATH. Audio extraction disabled.")
    if not SPEECH_REC_AVAILABLE:
        st.sidebar.warning("SpeechRecognition not found. Transcription disabled.")
    st.sidebar.success("All core libraries loaded.")
    st.sidebar.markdown("---")

    
    app_mode = st.sidebar.radio(
        "Select Module",
        ["Home", "Phase 1: EEG Analysis", "Phase 2: Behavioral Analysis", "Final Diagnosis"]
    )
    
    # ==================== HOME PAGE ====================
    if app_mode == "Home":
        st.title("Multimodal Mild Cognitive Impairment Detection using EEG and Behavioral Cues")
        st.markdown("""
        This application integrates three different data modalities to provide a comprehensive
        cognitive health assessment.
        
        ### 🧠 **Phase 1: EEG Analysis**
        Upload a `.set` EEG file to get a prediction based on brainwave activity.
        This module uses a pre-trained CatBoost model to analyze frequency band power.
        
        ### 🗣️ **Phase 2: Behavioral Analysis**
        Record and upload video responses to a series of cognitive prompts. This module
        extracts features from three sources:
        1.  **Visual:** Facial expressions and emotions (via DeepFace).
        2.  **Audio:** Voice characteristics like pauses, energy, and MFCCs (via Librosa).
        3.  **Text:** Transcribed speech features like filler words and verbal fluency.
        
        ### 🔬 **Final Diagnosis**
        This module performs a **weighted late-fusion** of the predictions from all
        three models (EEG, Behavioral-Visual, and Text) to produce a single,
        robust diagnostic percentage.
        
        **Instructions:**
        1.  Go to **Phase 1** and run an EEG analysis.
        2.  Go to **Phase 2** and run a full Behavioral analysis (Collect, Extract, Predict).
        3.  Go to **Final Diagnosis** to see the fused result.
        """)
        st.info("Ensure all required models (`.cbm`, `.joblib`) are in the correct directories.")

    # ==================== PHASE 1: EEG ANALYSIS ====================
    elif app_mode == "Phase 1: EEG Analysis":
        st.title("🧠 Phase 1: EEG Dementia Predictor")
        st.markdown("""
        Upload an EEG `.set` file (and its `.fdt` file if required).
        The model will predict **Healthy, Alzheimer's, or Frontotemporal Dementia**
        and save the probabilities for the final fusion.
        """)
        
        uploaded_files = st.file_uploader(
            "Upload EEG files (.set and optional .fdt)", 
            type=["set", "fdt"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            temp_paths = {}
            for f in uploaded_files:
                path = f.name
                with open(path, "wb") as out_file:
                    out_file.write(f.getbuffer())
                temp_paths[os.path.splitext(path)[1]] = path

            if ".set" not in temp_paths:
                st.error("❌ Please upload at least a .set file")
            else:
                eeg_file = temp_paths[".set"]
                try:
                    st.info("Extracting EEG features...")
                    progress_bar = st.progress(0)
                    raw_tmp = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
                    total_channels = len(raw_tmp.ch_names)
                    step = 1 / max(total_channels, 1)
                    progress = [0]

                    def update_progress():
                        progress[0] += step
                        progress_bar.progress(min(progress[0], 1.0))

                    features = extract_eeg_features(eeg_file, progress_callback=update_progress)
                    st.success("✅ Feature extraction complete!")
                    
                    st.subheader("📋 Extracted EEG Features (Sample)")
                   # Show all features with high precision (up to 12 decimal places)
                    features_list_df = pd.DataFrame(
                        [(k, f"{v:.12f}") for k, v in features.items()],
                        columns=["Feature", "Value"]
                    )

                    st.subheader("📋 Extracted EEG Features (High Precision List)")
                    st.dataframe(features_list_df, height=400)

                    features_wide_df = pd.DataFrame([features])
                    
                    st.info("Loading EEG model and predicting...")
                    if not os.path.exists(EEG_MODEL_FILE):
                        st.error(f"❌ Model file not found: {EEG_MODEL_FILE}")
                        return
                        
                    model = CatBoostClassifier()
                    model.load_model(EEG_MODEL_FILE)
                    
                    # Ensure columns match model's expected features
                    try:
                        model_features = model.get_feature_names()
                    except AttributeError:
                        # fallback if the CatBoost version doesn't support get_feature_names
                        model_features = features_wide_df.columns.tolist()

                    features_wide_df = features_wide_df.reindex(columns=model_features, fill_value=0)


                    # Get probabilities
                    pred_probs = model.predict_proba(features_wide_df)[0]
                    pred_numeric = int(np.argmax(pred_probs))
                    pred_label = EEG_DIAGNOSIS_MAP.get(pred_numeric, "Unknown")

                    st.subheader("🩺 EEG Predicted Diagnosis")
                    card_color = EEG_COLOR_MAP.get(pred_label, "#95a5a6")
                    st.markdown(f"""
                    <div style="background-color:{card_color}; padding:25px; border-radius:15px; text-align:center;
                                font-size:28px; font-weight:bold; color:white;">
                        {pred_label} (Confidence: {pred_probs[pred_numeric]:.1%})
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.bar_chart(pd.DataFrame({
                        'Class': [EEG_DIAGNOSIS_MAP.get(i, "N/A") for i in range(len(pred_probs))],
                        'Probability': pred_probs
                    }).set_index('Class'))

                    # Save probabilities for final fusion
                    eeg_probs_raw = {EEG_DIAGNOSIS_MAP[i]: prob for i, prob in enumerate(pred_probs)}
                    st.session_state['eeg_probs_raw'] = eeg_probs_raw
                    st.success("✅ EEG probabilities saved for Final Diagnosis!")

                except Exception as e:
                    st.error(f"❌ Error during EEG processing: {e}")
                finally:
                    for p in temp_paths.values():
                        if os.path.exists(p):
                            os.remove(p)

    # ==================== PHASE 2: BEHAVIORAL ANALYSIS ====================
    elif app_mode == "Phase 2: Behavioral Analysis":
        st.title("🗣️ Phase 2: Video-Based Behavioral Analysis")
        st.markdown("This module collects video data, extracts visual, audio, and text features, and trains/runs models on them.")
        
        if not all([DEEPFACE_AVAILABLE, SKLEARN_AVAILABLE, FFMPEG_AVAILABLE, SPEECH_REC_AVAILABLE]):
            st.error("One or more critical libraries (DeepFace, scikit-learn, SpeechRecognition) or executables (ffmpeg) are missing. This module may not function correctly.")

        sub_mode = st.radio("Select Task", 
                            ["Data Collection", "Feature Extraction", "Model Training", "Prediction"])
        
        st.markdown("---")

        # --- Phase 2: Data Collection ---
        if sub_mode == "Data Collection":
            st.header("📹 Video Recording Interface")
            subject_id = st.text_input("Enter Subject ID:", value="Subject_001")
            
            if subject_id:
                subject_dir = DATA_DIR / subject_id
                subject_dir.mkdir(exist_ok=True)
                
                prompt_idx = st.selectbox("Select Prompt", 
                                          range(len(COGNITIVE_PROMPTS)),
                                          format_func=lambda x: f"Prompt {x+1}: {COGNITIVE_PROMPTS[x]['text'][:50]}...")
                
                prompt = COGNITIVE_PROMPTS[prompt_idx]
                st.info(f"**Prompt {prompt['id']}:** {prompt['text']}")
                
                uploaded_file = st.file_uploader(
                    "Upload your video response (MP4, AVI, MOV, WEBM)", 
                    type=['mp4', 'avi', 'mov', 'webm'],
                    key=f"upload_{prompt_idx}"
                )
                
                if uploaded_file:
                    prompt_dir = subject_dir / f"prompt_{prompt['id']}"
                    prompt_dir.mkdir(exist_ok=True)
                    
                    video_path = prompt_dir / "response.mp4"
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_file.read())
                    st.success(f"✅ Video saved to: {video_path}")
                    
                    # --- NEW: Audio Extraction (using ffmpeg) ---
                    if FFMPEG_AVAILABLE:
                        try:
                            st.info("Extracting audio from video using ffmpeg...")
                            audio_path = prompt_dir / "audio.wav"
                            
                            # ffmpeg command:
                            # -i [input] -y [overwrite] -vn [no video] 
                            # -acodec pcm_s16le [codec] -ar 16000 [sample rate] -ac 1 [mono] [output]
                            command = [
                                "ffmpeg",
                                "-i", str(video_path),
                                "-y",  # Overwrite output file if it exists
                                "-vn", # No video
                                "-acodec", "pcm_s16le", # Match moviepy codec
                                "-ar", "16000", # Set sample rate for STT
                                "-ac", "1", # Set to mono for STT
                                str(audio_path)
                            ]
                            
                            # Run the command
                            result = subprocess.run(command, capture_output=True, text=True, check=True)

                            st.success(f"✅ Audio extracted to: {audio_path}")
                            st.audio(str(audio_path))
                            
                            # --- NEW: Speech-to-Text ---
                            if SPEECH_REC_AVAILABLE:
                                st.info("Transcribing audio to text (may take a moment)...")
                                r = sr.Recognizer()
                                with sr.AudioFile(str(audio_path)) as source:
                                    audio_data = r.record(source)
                                try:
                                    text = r.recognize_google(audio_data, language="en-US")
                                    text_path = prompt_dir / "transcription.txt"
                                    with open(text_path, 'w') as f:
                                        f.write(text)
                                    st.success("✅ Transcription complete!")
                                    st.text_area("Transcribed Text", text, height=100)
                                except sr.UnknownValueError:
                                    st.warning("Speech Recognition could not understand audio.")
                                except sr.RequestError as e:
                                    st.error(f"Could not request results from Google Speech Recognition; {e}")
                            else:
                                st.error("`speech_recognition` library not found. Skipping transcription.")
                        
                        except subprocess.CalledProcessError as e:
                            st.error(f"ffmpeg command failed with error:")
                            st.code(e.stderr)
                        except Exception as e:
                            st.error(f"Error during audio/text processing: {e}")
                    else:
                        st.error("`ffmpeg` executable not found. Skipping audio extraction.")

        # --- Phase 2: Feature Extraction ---
        elif sub_mode == "Feature Extraction":
            st.header("🔍 Feature Extraction")
            subjects = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
            if not subjects:
                st.warning("No subjects found. Please collect data first.")
                return
            
            selected_subject = st.selectbox("Select Subject", subjects)
            
            if st.button("Extract All Features"):
                st.info(f"Extracting features for {selected_subject}...")
                subject_dir = DATA_DIR / selected_subject
                progress_bar = st.progress(0)
                
                for i, prompt in enumerate(COGNITIVE_PROMPTS):
                    prompt_dir = subject_dir / f"prompt_{prompt['id']}"
                    if not prompt_dir.exists():
                        continue
                    
                    video_path = prompt_dir / "response.mp4"
                    audio_path = prompt_dir / "audio.wav"
                    
                    features = {}
                    
                    # 1. Visual Features
                    if video_path.exists():
                        features.update(extract_visual_features(video_path))
                    # 2. Audio (non-text) Features
                    if audio_path.exists():
                        features.update(extract_audio_features(audio_path))
                    # 3. Text Features
                    features.update(extract_text_features(prompt_dir, prompt['id']))
                    
                    features = convert_to_serializable(features)
                    feature_file = prompt_dir / "features.json"
                    with open(feature_file, 'w') as f:
                        json.dump(features, f, indent=2)
                    
                    progress_bar.progress((i + 1) / len(COGNITIVE_PROMPTS))
                
                all_features = aggregate_subject_features(subject_dir)
                feature_csv = subject_dir / f"{selected_subject}_features.csv"
                pd.DataFrame([all_features]).to_csv(feature_csv, index=False)
                
                st.success(f"✅ Feature extraction complete! Saved to {feature_csv}")
                st.dataframe(pd.DataFrame([all_features]).T.rename(columns={0: 'Value'}))

        # --- Phase 2: Model Training ---
        elif sub_mode == "Model Training":
            st.header("🤖 Model Training (3-Model Architecture)")
            st.markdown("""
            Upload a labels CSV file (Subject ID, Label). This will train **two separate models**:
            1.  **Behavioral Model:** Uses Visual (DeepFace) and Audio (Librosa) features.
            2.  **Text Model:** Uses features from the transcribed speech.
            (The EEG model is pre-trained)
            """)
            
            labels_file = st.file_uploader("Upload Labels CSV (e.g., `subject_id,label`)", type=['csv'])
            if labels_file:
                labels_df = pd.read_csv(labels_file)
                st.dataframe(labels_df.head())
                
                subject_col = st.selectbox("Select Subject ID Column", labels_df.columns, index=0)
                label_col = st.selectbox("Select Label Column", labels_df.columns, index=1)
                
                if st.button("Train Behavioral and Text Models"):
                    st.info("Loading features for all subjects...")
                    all_features_list = []
                    all_labels = []
                    
                    for _, row in labels_df.iterrows():
                        subject_id = str(row[subject_col]).strip()
                        label = str(row[label_col]).strip()
                        feature_file = DATA_DIR / subject_id / f"{subject_id}_features.csv"
                        
                        if feature_file.exists():
                            features = pd.read_csv(feature_file).iloc[0].to_dict()
                            all_features_list.append(features)
                            all_labels.append(label)
                    
                    if not all_features_list:
                        st.error("No feature files found! Please run 'Feature Extraction' first.")
                        return
                    
                    features_df = pd.DataFrame(all_features_list).fillna(0)
                    st.info(f"Training on {len(all_labels)} subjects. Classes: {set(all_labels)}")
                    
                    # Split features for the two models
                    behavioral_cols = [c for c in features_df.columns if c.startswith('vis_') or c.startswith('aud_') or c.startswith('temp_')]
                    text_cols = [c for c in features_df.columns if c.startswith('text_') or c.startswith('temp_')]
                    
                    X_beh = features_df[behavioral_cols]
                    X_text = features_df[text_cols]
                    y = np.array(all_labels)
                    
                    # Train Behavioral Model
                    st.write(f"Training Behavioral Model on {X_beh.shape[1]} features...")
                    model_beh, scaler_beh = train_rf_model(X_beh, y)
                    if model_beh:
                        joblib.dump(model_beh, BEHAVIORAL_MODEL_FILE)
                        joblib.dump(scaler_beh, BEHAVIORAL_SCALER_FILE)
                        st.success(f"✅ Behavioral Model saved to {BEHAVIORAL_MODEL_FILE}")
                    
                    # Train Text Model
                    st.write(f"Training Text Model on {X_text.shape[1]} features...")
                    model_text, scaler_text = train_rf_model(X_text, y)
                    if model_text:
                        joblib.dump(model_text, TEXT_MODEL_FILE)
                        joblib.dump(scaler_text, TEXT_SCALER_FILE)
                        st.success(f"✅ Text Model saved to {TEXT_MODEL_FILE}")

        # --- Phase 2: Prediction ---
        elif sub_mode == "Prediction":
            st.header("🎯 Behavioral & Text Prediction")
            
            if not all([BEHAVIORAL_MODEL_FILE.exists(), TEXT_MODEL_FILE.exists()]):
                st.error("Model files not found! Please train the models first.")
                return

            model_beh = joblib.load(BEHAVIORAL_MODEL_FILE)
            scaler_beh = joblib.load(BEHAVIORAL_SCALER_FILE)
            model_text = joblib.load(TEXT_MODEL_FILE)
            scaler_text = joblib.load(TEXT_SCALER_FILE)
            
            subjects = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
            selected_subject = st.selectbox("Select Subject for Prediction", subjects)
            
            if st.button("Run Behavioral & Text Prediction"):
                feature_file = DATA_DIR / selected_subject / f"{selected_subject}_features.csv"
                if not feature_file.exists():
                    st.error(f"Features for {selected_subject} not found! Please run 'Feature Extraction'.")
                    return
                
                features_df = pd.read_csv(feature_file)
                
                # --- Predict with Behavioral Model ---
                st.subheader(f"1. Behavioral (Video/Audio) Prediction")
                pred_beh, probs_beh = predict_cognitive_status(features_df, model_beh, scaler_beh)
                st.success(f"**Predicted Status:** {pred_beh}")
                
                prob_df_beh = pd.DataFrame({'Class': model_beh.classes_, 'Probability': probs_beh})
                st.bar_chart(prob_df_beh.set_index('Class'))
                
                beh_probs_raw = dict(zip(model_beh.classes_, probs_beh))
                st.session_state['behavioral_probs_raw'] = beh_probs_raw
                st.info("Behavioral probabilities saved for final fusion.")
                
                # --- Predict with Text Model ---
                st.subheader(f"2. Text (Transcription) Prediction")
                pred_text, probs_text = predict_cognitive_status(features_df, model_text, scaler_text)
                st.success(f"**Predicted Status:** {pred_text}")
                
                prob_df_text = pd.DataFrame({'Class': model_text.classes_, 'Probability': probs_text})
                st.bar_chart(prob_df_text.set_index('Class'))
                
                text_probs_raw = dict(zip(model_text.classes_, probs_text))
                st.session_state['text_probs_raw'] = text_probs_raw
                st.info("Text probabilities saved for final fusion.")


    # ==================== FINAL DIAGNOSIS (FUSION) ====================
    elif app_mode == "Final Diagnosis":
        st.title("🔬 Final Multimodal Diagnosis (Late-Fusion)")
        st.info("This page fuses results from EEG, Behavioral, and Text models.")

        eeg_ready = 'eeg_probs_raw' in st.session_state
        beh_ready = 'behavioral_probs_raw' in st.session_state
        text_ready = 'text_probs_raw' in st.session_state

        if not eeg_ready:
            st.warning("⚠️ Please run a prediction in 'Phase 1: EEG Analysis' first.")
        if not beh_ready or not text_ready:
            st.warning("⚠️ Please run a prediction in 'Phase 2: Behavioral Analysis' first.")
        
        if eeg_ready and beh_ready and text_ready:
            eeg_probs_raw = st.session_state['eeg_probs_raw']
            beh_probs_raw = st.session_state['behavioral_probs_raw']
            text_probs_raw = st.session_state['text_probs_raw']
            
            # --- 1. Process EEG Probs (Map to Healthy, MCI, Dementia) ---
            # EEG Map: {"Alzheimer's Disease", "Frontotemporal Dementia", "Healthy"}
            p_h_eeg = eeg_probs_raw.get("Healthy", 0.0)
            p_mci_eeg = 0.0  # EEG model does not predict MCI
            p_d_eeg = eeg_probs_raw.get("Alzheimer's Disease", 0.0) + eeg_probs_raw.get("Frontotemporal Dementia", 0.0)
            eeg_probs_final = np.array([p_h_eeg, p_mci_eeg, p_d_eeg])
            
            # --- 2. Process Behavioral Probs (Map to Healthy, MCI, Dementia) ---
            # Behavioral Map: (e.g., 'Healthy', 'MCI', 'AD')
            p_h_beh = beh_probs_raw.get("Healthy", 0.0)
            p_mci_beh = beh_probs_raw.get("MCI", 0.0)
            p_d_beh = beh_probs_raw.get("AD", 0.0) + beh_probs_raw.get("Dementia", 0.0)
            beh_probs_final = np.array([p_h_beh, p_mci_beh, p_d_beh])

            # --- 3. Process Text Probs (Map to Healthy, MCI, Dementia) ---
            # Text Map: (e.g., 'Healthy', 'MCI', 'AD')
            p_h_text = text_probs_raw.get("Healthy", 0.0)
            p_mci_text = text_probs_raw.get("MCI", 0.0)
            p_d_text = text_probs_raw.get("AD", 0.0) + text_probs_raw.get("Dementia", 0.0)
            text_probs_final = np.array([p_h_text, p_mci_text, p_d_text])
            
            st.subheader("Model Weights for Fusion")
            st.markdown("Adjust the influence of each model. (Weights will be normalized).")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                w_eeg = st.slider("EEG Weight (Default: High)", 0.0, 1.0, 0.5, 0.05)
            with col2:
                w_beh = st.slider("Behavioral (Video) Weight", 0.0, 1.0, 0.25, 0.05)
            with col3:
                w_text = st.slider("Text (Audio) Weight", 0.0, 1.0, 0.25, 0.05)
            
            # Normalize weights
            total_weight = w_eeg + w_beh + w_text
            if total_weight == 0: total_weight = 1 # Avoid division by zero
            
            w_eeg_norm = w_eeg / total_weight
            w_beh_norm = w_beh / total_weight
            w_text_norm = w_text / total_weight
            
            # --- 4. Perform Late Fusion ---
            fused_probs = (w_eeg_norm * eeg_probs_final) + \
                          (w_beh_norm * beh_probs_final) + \
                          (w_text_norm * text_probs_final)
            
            final_pred_idx = np.argmax(fused_probs)
            final_pred_label = FINAL_CLASSES[final_pred_idx]
            final_pred_percent = fused_probs[final_pred_idx]
            
            st.subheader("Final Fused Diagnosis")
            
            # Final metric card
            color = "#2ecc71" # Healthy
            if final_pred_label == "MCI":
                color = "#f1c40f" # Yellow
            elif final_pred_label == "Dementia":
                color = "#e74c3c" # Red
                
            st.markdown(f"""
            <div style="background-color:{color}; padding:30px; border-radius:15px; text-align:center;
                        font-size:32px; font-weight:bold; color:white;">
                {final_pred_label}
                <div style="font-size: 20px; font-weight: normal; margin-top: 10px;">
                (Final Confidence: {final_pred_percent:.1%})
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Final Probability Distribution")
            final_df = pd.DataFrame({
                'Class': FINAL_CLASSES,
                'Final Probability': fused_probs,
                'EEG (Normalized)': eeg_probs_final * w_eeg_norm,
                'Behavioral (Normalized)': beh_probs_final * w_beh_norm,
                'Text (Normalized)': text_probs_final * w_text_norm
            })
            
            st.bar_chart(final_df.set_index('Class')['Final Probability'])
            
            with st.expander("Show Detailed Probability Breakdown"):
                st.dataframe(final_df.set_index('Class'))


if __name__ == "__main__":
    # Check for new dependencies and warn user
    missing_deps = []
    if not SPEECH_REC_AVAILABLE:
        missing_deps.append("SpeechRecognition")
        missing_deps.append("PyAudio") # Often needed by SR
    
    if missing_deps:
        st.error(f"""
        **Missing Critical Dependencies!**
        
        This application requires new libraries for audio and text processing.
        Please install them by running:
        
        `pip install {" ".join(missing_deps)}`
        
        The application may crash without them.
        """)
    
    if not FFMPEG_AVAILABLE:
        st.error(f"""
        **Missing System Executable: ffmpeg**
        
        This application now uses `ffmpeg` for audio extraction.
        Please ensure `ffmpeg` is installed on your system and
        accessible in your environment's PATH.
        
        (e.g., run `brew install ffmpeg` or `apt-get install ffmpeg`)
        """)
    
    main()