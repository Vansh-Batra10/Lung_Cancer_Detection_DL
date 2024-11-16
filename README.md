---
---
# Lung Cancer Detection using Deep Learning

This project involves detecting lung cancer types using deep learning techniques. The classification task is divided into three categories: **Normal**, **Benign**, and **Malignant**. The project uses convolutional neural networks (CNNs) to classify medical images and explores multiple architectures, including advanced CNNs and ResNet50.

---

## Table of Contents
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Architectures](#model-architectures)
- [Performance](#performance)
- [Ensemble Learning](#ensemble-learning)
- [Results](#results)

---

## Dataset

The dataset is stored in the `DLPROJECTDATASET` directory, organized as follows:

```
DLPROJECTDATASET/
    ├── Normal/
    ├── Benign/
    └── Malignant/
```

Each subfolder contains image files corresponding to its class. The dataset is preprocessed and split into training, validation, and testing sets.

---

## Project Workflow

1. **Data Preprocessing**:
   - Images are loaded and labeled from the directory structure.
   - Data is split into training (70%), validation (15%), and test (15%) sets using `sklearn` utilities.
   - Preprocessing is done using ResNet preprocessing functions.

2. **Model Training**:
   - Multiple architectures, including Basic CNN, Advanced CNN, and ResNet50, are trained.
   - The models are trained using a categorical cross-entropy loss function with accuracy as the evaluation metric.

3. **Model Evaluation**:
   - Individual models are evaluated on the test set.
   - Metrics such as accuracy, precision, recall, and F1-score are calculated.

4. **Ensemble Learning**:
   - Predictions from all models are combined using an ensemble averaging approach for improved performance.

---

## Model Architectures

### Basic CNN
A lightweight CNN architecture with:
- Three convolutional layers
- Max pooling
- Fully connected layers with dropout for regularization

### Advanced CNN
A deeper CNN with:
- Five convolutional layers
- Larger filter sizes and more parameters
- Dropout and global average pooling for better generalization

### ResNet50-based Hybrid Model
A ResNet50 backbone used as a feature extractor with additional layers for fine-tuning.

---

## Performance

| Model          | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|-----------|--------|----------|
| Basic CNN       | 98.79%   | 98.83%    | 98.79% | 98.79%   |
| Advanced CNN    | 98.79%   | 98.80%    | 98.79% | 98.76%   |
| ResNet50 Hybrid | 96.36%   | 96.36%    | 96.36% | 96.36%   |
| **Ensemble**    | 99.39%   | 99.40%    | 99.39% | 99.39%   |

---

## Ensemble Learning

Ensemble predictions are made by averaging the outputs of all models, resulting in improved classification performance. The ensemble approach achieved the highest accuracy of **99.39%**.

---

## Results

### Classification Metrics
- **Basic CNN**: High accuracy with fewer parameters.
- **Advanced CNN**: Slightly better generalization than Basic CNN.
- **ResNet50 Hybrid**: Excellent performance as a feature extractor.
- **Ensemble**: Combines the strengths of all models for the best results.

### Confusion Matrix
Confusion matrices for each model provide detailed insights into the classification performance.

#### Basic CNN
```
[[17  0  1]
 [ 0 84  1]
 [ 0  0 62]]
```

#### Advanced CNN
```
[[16  1  1]
 [ 0 85  0]
 [ 0  0 62]]
```

#### ResNet50 Hybrid
```
[[15  0  3]
 [ 0 85  0]
 [ 3  0 59]]
```

#### Ensemble Model
```
[[17  0  1]
 [ 0 85  0]
 [ 0  0 62]]
```

--- 
