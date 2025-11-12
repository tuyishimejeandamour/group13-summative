# Multimodal Authentication & Product Recommendation System

## Project Overview

This project implements a User Identity and Product Recommendation System that uses multimodal authentication (facial recognition and voice verification) before providing personalized product recommendations.

**System Flow:**

1. User attempts to access the product prediction model
2. **Face Recognition**: System verifies user identity via facial recognition
3. If recognized, user proceeds to run a prediction
4. **Voice Verification**: System confirms the prediction through voiceprint verification
5. **Product Recommendation**: System displays the predicted product

The system uses pre-trained ML models to determine whether the face matches a known user, the voice is approved, and whether the prediction is allowed.

## Team

| Name                  | Task   | What They're Building                     |
| --------------------- | ------ | ----------------------------------------- |
| **Alice Mukarwema**   | Task 1 | Data merge + product recommendation model |
| **Yassin Hagenimana** | Task 2 | Image collection + facial recognition     |
| **Hirwa Brian**       | Task 3 | Audio collection + voice verification     |
| **Cedric Izabayo**    | Task 4 | Integration + system demo                 |

## Project Structure

**Note:** Empty folders contain README.md files describing what should go inside them.

```
multimodal-data-preprocessing-assignment/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                           # Download datasets here
│   │   ├── README.md
│   │   ├── customer_social_profiles.csv
│   │   └── customer_transactions.csv
│   ├── processed/                     # Merged/cleaned data
│   │   ├── README.md
│   │   └── merged_customer_data.csv
│   ├── images/                        # Facial images (3 per member)
│   │   ├── README.md
│   │   ├── alice/
│   │   ├── yassin/
│   │   ├── brian/
│   │   ├── cedric/
│   │   └── unauthorized/
│   ├── audio/                         # Voice recordings (2 per member)
│   │   ├── README.md
│   │   ├── alice/
│   │   ├── yassin/
│   │   ├── brian/
│   │   ├── cedric/
│   │   └── unauthorized/
│   └── features/                      # Extracted features
│       ├── README.md
│       ├── image_features.csv
│       └── audio_features.csv
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── README.md
│   ├── 01_data_merge_eda.ipynb
│   ├── 02_image_processing.ipynb
│   ├── 03_audio_processing.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/                               # Python source code
│   ├── data_processing/
│   │   ├── merge_data.py              # Data merging (Alice)
│   │   ├── image_processing.py        # Image preprocessing (Yassin)
│   │   └── audio_processing.py        # Audio preprocessing (Brian)
│   ├── models/
│   │   ├── product_recommender.py     # Product model (Alice)
│   │   ├── face_recognition.py        # Face model (Yassin)
│   │   └── voice_verification.py      # Voice model (Brian)
│   ├── pipeline/
│   │   └── authentication_pipeline.py # Integration (Cedric)
│   └── utils/
│       ├── evaluation.py              # Metrics helper
│       └── visualization.py           # Plotting helper
│
├── models/                            # Trained models (.pkl files)
│   ├── README.md
│   ├── product_recommender.pkl
│   ├── face_recognition.pkl
│   └── voice_verification.pkl
│
├── scripts/
│   ├── setup_directories.py           # Run this first
│   └── run_authentication_system.py   # Main CLI app (Cedric)
│
├── tests/                             # Test files
│   └── README.md
│
├── docs/                              # Additional documentation
│   └── README.md
│
└── reports/                           # Final deliverables
    ├── README.md
    ├── final_report.pdf
    ├── system_demo_video_link.txt
    └── team_contributions.md
```

## Setup

**You'll need:**

- Python 3.8+
- A webcam (for face photos)
- A microphone (for voice recordings)

**Installation:**

```bash
# Clone the repo
git clone <repo-url>
cd multimodal-data-preprocessing-assignment

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Create project folders
python scripts/setup_directories.py
```

## Data Collection (Everyone Needs to Do This!)

### Face Photos

Each person takes 3 photos with different expressions:

```
data/images/your_name/
├── neutral.jpg
├── smiling.jpg
└── surprised.jpg
```

**Tips:**

- Use your laptop webcam or phone camera
- Make sure your face is well-lit and clearly visible
- At least 640x480 resolution
- No sunglasses or masks

### Voice Recordings

Record yourself saying these two phrases:

```
data/audio/your_name/
├── yes_approve.wav
└── confirm_transaction.wav
```

**Tips:**

- WAV format preferred (16kHz or higher sample rate)
- 2-3 seconds per phrase
- Speak clearly, minimize background noise
- Use Voice Memos (Mac), Sound Recorder (Windows), or Audacity

## Task Breakdown

### Task 1: Data Merge & Product Recommendation (Alice)

**Where to work:** `notebooks/01_data_merge_eda.ipynb` and `src/data_processing/merge_data.py`

**What to do:**

1. Download these datasets:

   - [Customer Social Profiles](https://docs.google.com/spreadsheets/d/10up-WdC0a6egYaXLKiMQUotpOvaKZRZvYuWFQx4-RPQ/edit?gid=862127784#gid=862127784)
   - [Customer Transactions](https://docs.google.com/spreadsheets/d/1s4WOVm49lmLQ8d9QbbbdgAcRTNh3m0KCH_5ciiaRZw0/edit?gid=1844409263#gid=1844409263)

2. EDA - explore the data:

   - Summary stats
   - At least 3 plots with labels (distributions, correlations, etc.)
   - Check for missing values, duplicates, data types

3. Clean and merge:

   - Handle missing values
   - Fix data types
   - Merge the two datasets (document your approach)
   - Save to `data/processed/merged_customer_data.csv`

4. Build a product recommendation model:
   - Use XGBoost (recommended), Logistic Regression, or Random Forest
   - Calculate accuracy, F1-score, and loss
   - Save to `models/product_recommender.pkl`

### Task 2: Image Processing & Face Recognition (Yassin)

**Where to work:** `notebooks/02_image_processing.ipynb` and `src/data_processing/image_processing.py`

**What to do:**

1. Collect 3 face photos from each team member (12 total)

2. Load and display the images

3. Image augmentation - apply at least 2 techniques:

   - Rotation (±15 degrees)
   - Horizontal flip
   - Grayscale conversion
   - Brightness changes
   - Adding noise

4. Extract features from the images

   - Save to `data/features/image_features.csv`

5. Train a face recognition model:

   - Use XGBoost (recommended), Logistic Regression, or Random Forest
   - Calculate accuracy, F1-score, and loss
   - Save to `models/face_recognition.pkl`

6. Test with both authorized and unauthorized faces

### Task 3: Audio Processing & Voice Verification (Brian)

**Where to work:** `notebooks/03_audio_processing.ipynb` and `src/data_processing/audio_processing.py`

**What to do:**

1. Collect 2 voice recordings from each team member (8 total)

2. Load and visualize:

   - Load audio files
   - Show waveforms
   - Show spectrograms

3. Audio augmentation - apply at least 2 techniques:

   - Pitch shifting
   - Time stretching
   - Background noise
   - Volume adjustment
   - Speed changes

4. Extract audio features:

   - MFCCs (Mel-frequency cepstral coefficients)
   - Spectral roll-off
   - Energy
   - Zero-crossing rate
   - Save to `data/features/audio_features.csv`

5. Train a voice verification model:

   - Use XGBoost (recommended), Logistic Regression, or Random Forest
   - Calculate accuracy, F1-score, and loss
   - Save to `models/voice_verification.pkl`

6. Test with both authorized and unauthorized voices

### Task 4: Pipeline Integration (Cedric)

**Where to work:** `src/pipeline/authentication_pipeline.py` and `scripts/run_authentication_system.py`

**What to do:**

1. Integrate all three models:

   - Load the face recognition model
   - Load the voice verification model
   - Load the product recommendation model

2. Build the authentication flow:

   - Check face -> if authorized, allow access
   - Run product prediction
   - Check voice -> if authorized, show prediction
   - Handle rejections appropriately

3. Create a command-line app:

   - Takes face image as input
   - Verifies the person
   - Takes voice sample as input
   - Confirms the action
   - Shows product recommendation

4. Test everything:

   - Authorized users (should work)
   - Unauthorized face (should block)
   - Unauthorized voice (should block)

5. Record a demo video (3-5 minutes):
   - Show an unauthorized face being rejected
   - Show an unauthorized voice being rejected
   - Show a successful full authentication with product display

## Git Workflow

**Branch names:**

```
feature/data-merge              (Alice)
feature/image-processing        (Yassin)
feature/audio-processing        (Brian)
feature/pipeline-integration    (Cedric)
```

**Daily routine:**

```bash
# Pull latest changes
git checkout main
git pull origin main

# Create/switch to your branch
git checkout -b feature/your-task-name

# Do your work, then commit
git add .
git commit -m "Describe what you did"

# Push to GitHub
git push origin feature/your-task-name

# Create a Pull Request when ready
```

## Model Requirements

**All models must:**

- Use XGBoost (recommended), Logistic Regression, or Random Forest
- Report accuracy, F1-score, and loss

**Quick evaluation example:**

```python
from sklearn.metrics import accuracy_score, f1_score, log_loss

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
loss = log_loss(y_true, y_pred_proba)
```

**Saving/loading models:**

```python
import joblib
joblib.dump(model, 'models/model_name.pkl')
model = joblib.load('models/model_name.pkl')
```

## What to Submit

**GitHub repo with:**

- [ ] All code files
- [ ] Data files (raw, processed, features)
- [ ] Trained models (.pkl files)
- [ ] Jupyter notebooks
- [ ] This README

**Required files:**

- [ ] `data/processed/merged_customer_data.csv`
- [ ] `data/features/image_features.csv`
- [ ] `data/features/audio_features.csv`
- [ ] `models/product_recommender.pkl`
- [ ] `models/face_recognition.pkl`
- [ ] `models/voice_verification.pkl`
- [ ] `scripts/run_authentication_system.py`

**Final report (PDF) covering:**

- [ ] Your approach for each task
- [ ] Data preprocessing steps
- [ ] Model training process
- [ ] Evaluation results
- [ ] Challenges you faced
- [ ] Each person's contributions

**Demo video (3-5 minutes):**

- [ ] Link saved in `reports/system_demo_video_link.txt`
- [ ] Show unauthorized face being rejected
- [ ] Show unauthorized voice being rejected
- [ ] Show complete successful authentication

## Demo Video

**Show these scenarios:**

1. Unauthorized face -> "Access Denied"
2. Unauthorized voice -> "Access Denied"
3. Full successful flow:
   - Authorized face -> Access granted
   - Product prediction runs
   - Authorized voice -> Prediction confirmed
   - Product recommendation displayed

**Video format:**

- 3-5 minutes
- Screen recording with voiceover
- Upload to YouTube or Google Drive
- Put link in `reports/system_demo_video_link.txt`

## Grading (40 points total)

| Component                     | Points |
| ----------------------------- | ------ |
| EDA quality & insights        | 4      |
| Data cleaning & merge         | 4      |
| Image collection & diversity  | 4      |
| Image augmentation & features | 4      |
| Audio quality & visualization | 4      |
| Audio augmentation & features | 4      |
| Model implementation          | 4      |
| Evaluation & integration      | 4      |
| System demo                   | 4      |
| Submission quality            | 4      |

## Resources

**Image Processing:**

- [OpenCV Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Pillow Docs](https://pillow.readthedocs.io/)

**Audio Processing:**

- [Librosa Docs](https://librosa.org/doc/latest/index.html)

**Machine Learning:**

- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/)

**Data:**

- [Pandas](https://pandas.pydata.org/docs/)

## Important Dates

- **Deadline:** November 14, 2025 @ 11:59 PM
- **Submit:** PDF report via Canvas
- **Attempts:** 2 allowed

---

Happy hacking! 🚀
