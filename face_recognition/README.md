# Face Recognition Model

This folder contains all resources related to the face recognition model.

## Structure

```
face_recognition/
├── data/
│   ├── images/              # Facial images (3 per member: neutral, smiling, surprised)
│   ├── image_features.csv   # Extracted image features
│   ├── face_recognition_model.pkl
│   ├── face_recognition_scaler.pkl
│   └── face_recognition_metadata.json
├── src/
│   ├── image_processing.py      # Image preprocessing and feature extraction
│   ├── predict_face.py          # Face recognition predictor
│   ├── batch_predict_face.py    # Batch prediction script
│   └── display_sample_images.py # Display sample images
└── face_recognition_model.ipynb  # Jupyter notebook for model training
```

## Model Details

- **Task**: Task 2 - Image collection + facial recognition
- **Developer**: Yassin Hagenimana
- **Model Type**: Random Forest with VGG16 feature extraction
- **Input**: Facial images (224x224)
- **Output**: User identification with confidence score

## Usage

See the notebook `face_recognition_model.ipynb` for training and evaluation.

