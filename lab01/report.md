# Lab 1: Environment and First System Measurements

## 1. Goal

The goal of this lab was to create a reproducible Python environment, train two baseline machine learning models, measure their system costs, and evaluate their suitability for different deployment environments.

The two models used were Logistic Regression and Random Forest Classifier on the Breast Cancer Wisconsin dataset.

## 2. Method

Python 3.11.8 was used inside an isolated virtual environment.

The Breast Cancer Wisconsin dataset was loaded using scikit-learn.

The dataset was divided into 70% training data and 30% test data using a stratified split with random_state=42.

The models were:

- LogisticRegression(max_iter=1000, random_state=42)
- RandomForestClassifier(n_estimators=100, random_state=42)

Test accuracy was measured for both models.

Training time was measured after one warm-up run and repeated five times. The median training time was reported.

Single-sample inference was measured after a warm-up and repeated 100 times. The median inference latency was reported.

Each trained model was saved using joblib and its file size was measured.

Peak process memory was measured using psutil during training and inference.

## 3. Results

### Accuracy

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 0.9415 |
| Random Forest Classifier | 0.9357 |

### System Measurements

| Model | Training Median (ms) | Inference Median (ms) | Model Size (KB) | Peak Training Memory (MB) | Peak Inference Memory (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 306.2526 | 0.0447 | 1.03 | 102.10 | 102.11 |
| Random Forest Classifier | 98.5617 | 1.8050 | 284.07 | 103.58 | 104.21 |

### Deployment Suitability

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | Yes | Yes | Yes | No |
| Random Forest Classifier | Yes | Yes | Yes | No |

Both models have inference latency and model size small enough for Cloud, Edge, and Mobile deployment.

Logistic Regression has an extremely small model size of 1.03 KB and very low inference latency. However, the measured Python process memory usage was approximately 102 MB, which is far above the TinyML memory budget of 256 KB.

Random Forest also satisfies the Cloud, Edge, and Mobile latency and size requirements. However, its model size is approximately 284.07 KB, which exceeds the TinyML model-size budget of 100 KB. Its measured memory usage also exceeds the TinyML memory budget.

Therefore, neither model is suitable for TinyML in the measured Python environment.

## 4. Conclusions

1. Logistic Regression achieved slightly higher test accuracy than Random Forest while also producing a much smaller model file.

2. Random Forest trained faster in these measurements, but its single-sample inference latency and model size were considerably larger than Logistic Regression.

3. Both models are suitable for Cloud, Edge, and Mobile deployment based on the measured latency, memory, and model size, but neither model satisfies the TinyML resource limits.