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
| **denise**   | Task 1 | Data merge + product recommendation model |
| **damour** | Task 2 | Image collection + facial recognition     |
| **kellia**       | Task 3 | Audio collection + voice verification     |
| **stecie**    | Task 4 | Integration + system demo                 |

**Note:** Current implementation includes face recognition module with team members: damour, denise, kelia, and stecie.

## Project Structure

```
group13/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── face_recognition/                  # Face Recognition Module (Task 2)
│   ├── README.md                      # Face recognition documentation
│   ├── face_recognition_model.ipynb   # Model training notebook
│   ├── data/                          # Facial images (3 per member)
│   │   ├── damour/
│   │   ├── denise/
│   │   ├── kelia/
│   │   └── stecie/
│   └── src/                           # Face recognition source code
│       ├── image_processing.py        # Image preprocessing & feature extraction
│       ├── predict_face.py            # Single image prediction
│       ├── batch_predict_face.py      # Batch prediction
│       └── display_sample_images.py   # Display sample images
│
├── voice_recognition/                 # Voice Recognition Module (Task 3)
│   └── README.md                      # Voice recognition documentation
│
├── product_recommender/               # Product Recommendation Module (Task 1)
│   └── README.md                      # Product recommender documentation
│
├── src/                               # Shared Python source code
│   └── config.py                      # Centralized configuration module
│
├── data/                              # Project data (if exists)
│   ├── features/                      # Extracted features
│   │   └── image_features.csv         # Image features from face recognition
│   └── processed/                     # Processed datasets
│
└── models/                            # Trained models (.pkl files)
    ├── face_recognition_model.pkl     # Trained face recognition model
    ├── face_recognition_scaler.pkl    # Feature scaler
    └── face_recognition_metadata.json # Model metadata
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
cd group13

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# For face recognition, you'll also need TensorFlow
pip install tensorflow>=2.13.0
```

**Quick Start - Face Recognition:**

```bash
# Navigate to face recognition module
cd face_recognition

# Run the training notebook
jupyter notebook face_recognition_model.ipynb

# Or use the Python scripts directly
cd src
python image_processing.py
python predict_face.py ../data/denise/neutral.jpeg
```

## Data Collection (Everyone Needs to Do This!)

### Face Photos

Each person takes 3 photos with different expressions:

```
face_recognition/data/your_name/
├── neutral.jpeg (or .jpg)
├── smiling.jpeg (or .jpg)
└── surprised.jpeg (or .jpg)
```

**Current team members:** damour, denise, kelia, stecie

**Tips:**

- Use your laptop webcam or phone camera
- Make sure your face is well-lit and clearly visible
- At least 640x480 resolution
- No sunglasses or masks

### Voice Recordings

Record yourself saying these two phrases:

```
voice_recognition/data/your_name/
├── yes_approve.wav
└── confirm_transaction.wav
```

