# Product Recommendation Model

Advanced product recommendation system that predicts product categories based on customer behavior, engagement metrics, and transaction history.

## 📊 Model Performance

- **F1 Score**: 0.43
- **Algorithm**: Random Forest Classifier
- **Classes**: Books, Clothing, Electronics, Groceries, Sports

## 🚀 Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn imbalanced-learn
```

### Load Model

```python
import joblib
import pandas as pd
import numpy as np

# Load artifacts
model = joblib.load('./best_models/product_recommendation_model.pkl')
scaler = joblib.load('./best_models/scaler.pkl')
label_encoder = joblib.load('./best_models/label_encoder.pkl')

print(f"Available categories: {label_encoder.classes_}")
```

### Make Prediction

```python
# Prepare customer data (31 features required)
customer_data = pd.DataFrame({
    'purchase_amount': [150.50],
    'customer_rating': [4.5],
    'engagement_score': [75.0],
    'purchase_interest_score': [8.5],
    'purchase_month': [6],
    'purchase_day_of_week': [2],
    'purchase_day_of_month': [15],
    'purchase_quarter': [2],
    'is_weekend': [0],
    'is_month_end': [0],
    'purchase_amount_count': [10],
    'purchase_amount_mean': [145.25],
    'purchase_amount_sum': [1452.50],
    'purchase_amount_std': [25.30],
    'purchase_amount_min': [50.00],
    'purchase_amount_max': [300.00],
    'customer_rating_mean': [4.3],
    'customer_rating_std': [0.5],
    'customer_rating_min': [3.0],
    'customer_rating_max': [5.0],
    'engagement_score_mean': [72.5],
    'engagement_score_std': [8.2],
    'engagement_score_min': [55.0],
    'engagement_score_max': [90.0],
    'purchase_interest_score_mean': [8.2],
    'purchase_interest_score_std': [1.1],
    'amount_rating_interaction': [677.25],
    'engagement_interest_interaction': [637.5],
    'amount_per_engagement': [1.98],
    'platform_frequency': [150],
    'sentiment_numeric': [2]
})

# Scale and predict
X_scaled = scaler.transform(customer_data)
prediction = model.predict(X_scaled)
category = label_encoder.inverse_transform(prediction)

print(f"Predicted category: {category[0]}")

# Get top 3 recommendations with confidence
if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(X_scaled)[0]
    top_3_idx = np.argsort(probs)[-3:][::-1]
    
    for idx in top_3_idx:
        product = label_encoder.inverse_transform([idx])[0]
        confidence = probs[idx]
        print(f"{product}: {confidence:.2%}")
```

## 📋 Required Features (31 Total)

### Basic Transaction (4)
- `purchase_amount`, `customer_rating`, `engagement_score`, `purchase_interest_score`

### Temporal (6)
- `purchase_month`, `purchase_day_of_week`, `purchase_day_of_month`, `purchase_quarter`, `is_weekend`, `is_month_end`

### Customer Behavior (16)
- `purchase_amount_*` (count, mean, sum, std, min, max)
- `customer_rating_*` (mean, std, min, max)
- `engagement_score_*` (mean, std, min, max)
- `purchase_interest_score_*` (mean, std)

### Interactions (3)
- `amount_rating_interaction` = purchase_amount × customer_rating
- `engagement_interest_interaction` = engagement_score × purchase_interest_score
- `amount_per_engagement` = purchase_amount / (engagement_score + 1)

### Platform & Sentiment (2)
- `platform_frequency`, `sentiment_numeric` (2=Positive, 1=Neutral, 0=Negative)

## 📈 Model Details

**Top 5 Important Features:**
1. `purchase_day_of_month` (7.95%)
2. `amount_rating_interaction` (4.91%)
3. `amount_per_engagement` (4.63%)
4. `engagement_score_mean` (4.55%)
5. `purchase_amount` (4.42%)

**Training:**
- Dataset: 140 samples after cleaning
- Split: 75% train, 25% test
- Preprocessing: RobustScaler, SMOTE (if available)
- Feature Selection: SelectKBest + RFE (26/31 features used)

**Per-Class Performance:**
- Books: F1=0.50
- Clothing: F1=0.50
- Electronics: F1=0.22
- Groceries: F1=0.57
- Sports: F1=0.33

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ValueError: X has 20 features, but model expects 26` | Include all 31 required features |
| Wrong predictions | Ensure features are in correct order |
| Random outputs | Use `scaler.transform()` before prediction |
| `ModuleNotFoundError: xgboost` | `pip install xgboost` or use Random Forest fallback |

## 🔄 Data Pipeline

```bash
# 1. Merge datasets
jupyter notebook merge_dataset.ipynb

# 2. Train model
jupyter notebook product_recommendation_model.ipynb

# 3. Test predictions
jupyter notebook test_product_recommendation.ipynb
```

## 📝 Files Description

- **`merge_dataset.ipynb`**: Merges customer_transactions + customer_social_profiles
- **`product_recommendation_model.ipynb`**: Feature engineering, training, optimization
- **`test_product_recommendation.ipynb`**: Example predictions and testing
- **`best_models/`**: Saved model artifacts (pkl files + metadata)