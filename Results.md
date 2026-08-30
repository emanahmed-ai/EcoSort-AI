# EcoSort AI — Experimental Results & Evaluation Report

This document details the quantitative performance, evaluation curves, and cross-validation metrics across all evaluated architectures within the **EcoSort AI** framework.

---

## 🏆 Summary of Model Performances

All models were evaluated on the verified, held-out test split (1,226 images) under strict zero-data-leakage protocols.

| Metric | ConvNeXt-Tiny (Primary) | EfficientNetV2-S | MobileNetV2 (Edge) |
| :--- | :---: | :---: | :---: |
| **Primary Deployment Focus** | High Feature Accuracy | Balanced Efficiency | Edge Inference |
| **Test Accuracy** | **96.33%** | **94.8%** | **94.21%** |
| **Macro Precision** | 95.74% | 95.00% | 93.53% |
| **Macro Recall** | 95.54% | 94.94% | 92.91% |
| **Macro F1-Score** | 95.60% | 94.95% | 93.14% |


---

## 🔬 Performance Visualizations( ConvNeXt-Tiny)

### Confusion Matrix
Confusion Matrix
<img width="1800" height="1400" alt="confusion_matrix" src="https://github.com/user-attachments/assets/9dcd588a-4b2f-4046-94d4-e567cde69c41" />

### Training & Validation Loss/Accuracy Curves ( ConvNeXt-Tiny)
Training Curves
<img width="2800" height="1000" alt="training_curves" src="https://github.com/user-attachments/assets/ff37bd7c-7a97-4161-9d43-7d7c3e16b763" />


---

## 🔬 ConvNeXt-Tiny In-Depth Evaluation

### Generalization & Split Analysis
- **Training Accuracy:** 99.65%
- **Validation Accuracy (5-Fold CV):** 95.49%
- **Final Test Accuracy:** 96.33%
- **Diagnosis:** Well Generalized. The minimal gap between 5-fold CV validation ($95.49\%$) and held-out test performance ($96.33\%$) confirms strong robustness against overfitting without domain leakage.

---

## 📈 Key Technical Takeaways

1. **ConvNeXt-Tiny Leadership:** Modern convolutional blocks combined with $384 \times 384$ input resolution provided superior feature extraction capability for fine-grained material differentiation.
2. **MobileNetV2 Edge Efficiency:** Achieved **94.21% test accuracy** while maintaining a lightweight footprint, making it the ideal candidate for resource-constrained edge hardware or smart bin integrations.
3. **Domain Shift & Real-World Complexity:** High dataset precision provides a solid benchmark; ongoing work focuses on testing under severe occlusion, dirt, overlapping items, and variable lighting conditions
