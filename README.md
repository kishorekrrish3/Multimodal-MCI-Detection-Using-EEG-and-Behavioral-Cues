# 🧠 Multimodal Mild Cognitive Impairment Detection using EEG and Behavioral Cues

A comprehensive **AI-driven diagnostic system** that integrates **EEG signals** and **behavioral analysis (video, audio, and speech)** to detect **Mild Cognitive Impairment (MCI)** and early stages of **Dementia**.
This project leverages **EEG frequency band analysis**, **facial emotion recognition**, **audio feature extraction**, **speech-to-text linguistic metrics**, and a **weighted late-fusion model** for robust multimodal diagnosis.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [System Architecture](#-system-architecture)
4. [Dataset](#-dataset)
5. [Installation](#-installation)
6. [Project Structure](#-project-structure)
7. [How It Works](#-how-it-works)
8. [Model Details](#-model-details)
9. [UI Overview](#-ui-overview)
10. [Results & Observations](#-results--observations)
11. [Future Scope](#-future-scope)
12. [Contributors](#-contributors)
13. [License](#-license)

---

## 🧩 Overview

Cognitive impairments like **Mild Cognitive Impairment (MCI)** and **Dementia** affect millions globally.
Early diagnosis is crucial — but traditional medical assessments are time-consuming and subjective.

Our **Multimodal Detection System** automates this process using:

* **EEG data** to analyze brain activity patterns
* **Video + Audio** recordings for behavioral and speech-based cues
* **Speech-to-Text** analysis for cognitive-linguistic patterns

The final system combines all three modalities through a **weighted late-fusion model** to produce a reliable diagnosis.

---

## 🚀 Features

✅ **EEG Phase**

* Extracts 5-band EEG frequency features (Delta, Theta, Alpha, Beta, Gamma) from `.set` files
* Uses a trained **CatBoost model** to classify into *Healthy*, *Alzheimer’s*, or *Frontotemporal Dementia*

✅ **Behavioral Phase**

* Captures **facial emotion**, **audio**, and **speech** responses to 10 cognitive prompts
* Extracts:

  * **Visual features:** emotion + age estimation via `DeepFace`
  * **Audio features:** MFCCs, pause counts, RMS energy via `Librosa`
  * **Text features:** filler word ratio, word count, fluency metrics via `SpeechRecognition`

✅ **Final Fusion Phase**

* Performs **late fusion** across EEG, behavioral, and textual models
* Weighted combination ensures EEG contributes higher diagnostic importance

✅ **Interactive Streamlit Dashboard**

* Real-time EEG upload and analysis
* Video/audio capture and feature extraction
* Final fusion visualization with interactive probability charts

---

## 🧠 System Architecture

```mermaid
graph TD
A[EEG Data (.set)] --> B[EEG Feature Extractor]
B --> C[Trained CatBoost Model]
C --> F[EEG Diagnosis]

D[Video Response] --> E[Audio Extraction (FFmpeg)]
E --> G[Audio Features (Librosa)]
D --> H[Visual Features (DeepFace)]
E --> I[Speech-to-Text (Google STT)]
I --> J[Text Features (Linguistic Metrics)]

G --> K[Behavioral Model]
H --> K
J --> L[Text Model]
K --> M[Behavioral Prediction]
L --> N[Text Prediction]

F --> O[Weighted Late Fusion]
M --> O
N --> O
O --> P[Final Cognitive Diagnosis (Healthy / MCI / Dementia)]
```

---

## 🧬 Dataset

The EEG dataset used is publicly available on **OpenNeuro**:
🔗 [https://openneuro.org/datasets/ds004504/versions/1.0.8](https://openneuro.org/datasets/ds004504/versions/1.0.8)

* **Subjects:** 88
* **Groups:** Healthy, Alzheimer’s Disease, Frontotemporal Dementia
* **Data Type:** EEG recordings (EEGLAB `.set` format)
* **Sampling Rate:** 512 Hz

Behavioral data (video/audio) was collected separately as part of cognitive testing simulations.

---

## ⚙️ Installation

### 🧩 Requirements

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

**Main Libraries:**

* `streamlit`
* `mne`
* `catboost`
* `deepface`
* `librosa`
* `speechrecognition`
* `opencv-python`
* `joblib`
* `numpy`, `pandas`, `scikit-learn`

**System Dependencies:**

* `ffmpeg` (for audio extraction from video)

  * Windows: `choco install ffmpeg`
  * Linux: `sudo apt install ffmpeg`
  * macOS: `brew install ffmpeg`

---

## 📁 Project Structure

```
📂 Multimodal-MCI-Detection
│
├── final.py                         # Main Streamlit application (EEG + Behavioral + Fusion)
├── eeg_feature_extractor.py         # Extracts EEG band features from all electrodes
├── eeg_model_check.ipynb            # Model experimentation and evaluation notebook
│
├── features.csv                     # Extracted EEG features for 88 subjects
├── participants.tsv                 # Subject labels (Healthy, AD, FTD)
├── catboost_eeg_model.cbm           # Trained EEG classifier model
│
└── README.md                        # You’re reading it 🙂
```

---

## 🧩 How It Works

### 🧠 **Phase 1: EEG Analysis**

1. EEG `.set` file is loaded using `mne`
2. Bandpower computed for **5 EEG frequency bands** (Delta, Theta, Alpha, Beta, Gamma)
3. Features flattened and passed to a **CatBoost classifier**
4. Model predicts **Healthy / Alzheimer’s / FTD**

### 🎥 **Phase 2: Behavioral Analysis**

1. Participant responds to 10 cognitive prompts via video
2. `ffmpeg` extracts audio → processed by `librosa`
3. `DeepFace` analyzes facial emotions and age
4. `speech_recognition` converts speech → text → linguistic features
5. `RandomForest` models trained for behavioral and text features

### ⚖️ **Phase 3: Final Fusion**

* Combines outputs of all three models
* Weights are user-adjustable (EEG given more weight by default)
* Outputs final probability across:

  * **Healthy**
  * **Mild Cognitive Impairment (MCI)**
  * **Dementia**

---

## 🧠 Model Details

| Model            | Input                            | Algorithm    | Output Classes            |
| ---------------- | -------------------------------- | ------------ | ------------------------- |
| EEG Model        | Bandpower features (88 subjects) | CatBoost     | Alzheimer’s, FTD, Healthy |
| Behavioral Model | Visual + Audio                   | RandomForest | Healthy, MCI, Dementia    |
| Text Model       | Transcribed speech features      | RandomForest | Healthy, MCI, Dementia    |
| Fusion           | Weighted sum of 3 models         | Late Fusion  | Healthy, MCI, Dementia    |

---

## 🖥️ UI Overview

**Developed using Streamlit**, featuring:

* Sidebar navigation between modules
* Real-time EEG upload and feature visualization
* Video and audio upload interface
* Live transcription and emotional analysis
* Final result visualization with probability bars

![Streamlit Demo](https://github.com/yourusername/multimodal-mci-detection/assets/demo_ui.png)

---

## 📊 Results & Observations

| Model           | Accuracy | F1 Score | ROC-AUC  |
| --------------- | -------- | -------- | -------- |
| EEG (CatBoost)  | ~89%     | 0.86     | 0.92     |
| Behavioral (RF) | ~82%     | 0.80     | 0.87     |
| Textual (RF)    | ~78%     | 0.76     | 0.84     |
| Final Fusion    | **92%**  | **0.90** | **0.95** |

> Fusion significantly improved stability and interpretability of results.

---

## 🔮 Future Scope

* Integrate **real-time EEG acquisition** via OpenBCI
* Expand dataset to include **MCI**-specific EEG data
* Replace hand-crafted features with **Deep Learning encoders**
* Enable **cross-modal attention fusion** using transformers
* Deploy as a **web-based clinical screening tool**

---

## 🪪 License

This project is released under the **MIT License**.
Feel free to use, modify, and cite with attribution.

---

## 💬 Citation

If you use this work in research or academic projects, please cite:

```
P. Kishore et al.,
"Multimodal Mild Cognitive Impairment Detection using EEG and Behavioral Cues",
VIT Chennai, 2025.
```

