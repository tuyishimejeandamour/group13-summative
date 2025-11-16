# Voice Recognition Module

This module processes voice recognition data and builds voiceprint verification models for the project.

## Setup

### Prerequisites

1. **Install Python packages:**
   ```bash
   pip install librosa soundfile matplotlib pandas numpy scipy scikit-learn xgboost joblib seaborn
   ```

## Usage

### Step 1: Process Voice Data

Open and run the first notebook to extract features from audio files:

```bash
jupyter notebook process_voice_data.ipynb
```

This notebook will:
1. Load all WAV audio samples from the `data/` directory
2. Display waveforms and spectrograms for each sample
3. Apply augmentations (pitch shift, time stretch, background noise)
4. Extract features (MFCCs, spectral roll-off, energy, etc.)
5. Save features to `features/audio_features.csv`
6. Save visualizations to `reports/`

### Step 2: Train Voiceprint Verification Model

After processing the voice data, train the verification model:

```bash
jupyter notebook voiceprint_verification_model.ipynb
```

This notebook will:
1. Load extracted features from `features/audio_features.csv`
2. Train multiple models (Random Forest, Logistic Regression, XGBoost)
3. Evaluate models using Accuracy, F1-Score, and Loss metrics
4. Compare model performance with visualizations
5. Save the best model to `models/` directory
6. Generate confusion matrix and feature importance plots

## Project Structure

```
voice_recognition/
├── data/                    # Audio files (WAV format)
│   ├── damour/
│   ├── denise/
│   ├── kelia/
│   └── stecie/
├── src/                     # Source code modules
│   ├── audio_loader.py      # Audio loading utilities
│   ├── audio_visualization.py # Visualization functions
│   ├── audio_augmentation.py # Augmentation functions
│   └── feature_extraction.py # Feature extraction
├── features/                # Extracted features (CSV)
│   └── audio_features.csv   # Feature dataset
├── models/                  # Trained models
│   ├── voiceprint_verification_model.pkl
│   ├── voiceprint_scaler.pkl
│   ├── voiceprint_label_encoder.pkl
│   └── voiceprint_model_metadata.json
├── reports/                 # Visualization outputs
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── [audio visualizations]
├── process_voice_data.ipynb # Step 1: Feature extraction notebook
└── voiceprint_verification_model.ipynb # Step 2: Model training notebook
```

## Workflow

1. **Data Processing** (`process_voice_data.ipynb`):
   - Load audio files from `data/` directory
   - Visualize waveforms and spectrograms
   - Apply data augmentations
   - Extract audio features
   - Save features to CSV

2. **Model Training** (`voiceprint_verification_model.ipynb`):
   - Load extracted features
   - Train and compare multiple ML models
   - Evaluate performance metrics
   - Save best model for deployment

## Output Files

- `features/audio_features.csv`: Extracted audio features for all samples
- `models/voiceprint_verification_model.pkl`: Trained model (best performing)
- `models/voiceprint_scaler.pkl`: Feature scaler for preprocessing
- `models/voiceprint_label_encoder.pkl`: Label encoder for predictions
- `models/voiceprint_model_metadata.json`: Model metadata and performance metrics
- `reports/`: Visualizations including model comparisons, confusion matrices, and feature importance
