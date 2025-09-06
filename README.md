# ROP Multitask EfficientNet-B0

This project is a multi-task deep learning model for the classification of **Retinopathy of Prematurity (ROP)**.
The model is based on EfficientNet-B0 and predicts two outcomes simultaneously:

- **DG (Disease Grade)** classification
- **PF (Plus Disease)** classification

---

## 🚀 Features
- **EfficientNet-B0 backbone** (ImageNet pre-trained)
- **Multi-task learning** (separate head layers for DG and PF)
- **Data augmentation** (RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter)
- **Class balancing** (WeightedRandomSampler + Class Weights)
- **Accelerated training with Mixed Precision Training (AMP)**
- **Learning rate planner with CosineAnnealingLR**

---

## 📂 Data Structure
The dataset consists of a CSV and an image folder:

Data "Retinal Image Dataset of Infants and ROP" Taken from Kaggle.

Results

Sample results (after 10 epochs):

- **DG: Train F1 ≈ 0.96 | Val F1 ≈ 0.44

- **PF: Train F1 ≈ 0.99 | Val F1 ≈ 0.88

Strong success was observed in the PF class, while more difficult classification was observed in the DG.

🔮 Future Work

- **Increasing the diversity of data augmentation to improve DG performance

- **Experimenting with more powerful backbone models (EfficientNet-B3, Swin Transformer, etc.)

- **Transfer learning + fine-tuning strategies

- **Better class imbalance methods (SMOTE, focal loss)
