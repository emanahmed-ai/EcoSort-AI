# EcoSort AI: Deep Learning-Based Garbage Classification

EcoSort AI 
<img width="1586" height="992" alt="option-1 (1)" src="https://github.com/user-attachments/assets/87aa78bf-1ddb-4d5e-aa53-7bc042b14f30" />


**EcoSort AI** is an end-to-end deep learning framework designed to enable reliable, real-world garbage classification. While standard waste datasets often achieve high accuracy under controlled conditions (clean objects, uniform backgrounds, single items), real-world waste recognition faces severe challenges such as occlusion, dirty or damaged materials, overlapping items, and complex backgrounds. 

## 👥 Project Team & Supervision

* **Supervised by:** Professor Dr. Mohamed El-Haddad
* **Team Members:**
  * Asmaa Eldaly
  * Asmaa Salah
  * Aya Abdelnaem
  * Eman Ahmed
  * Amira Salama

This project establishes a strict, leak-free computer vision pipeline evaluating lightweight and modern convolutional architectures (**ConvNeXt-Tiny**, **MobileNetV2**, and **EfficientNetV2-S**) to bridge the gap between dataset accuracy and practical deployment.
---

## 📌 Project Features & Highlights

- **Data Integrity & Leak-Free Design:** Strict separation between training, validation, and held-out test splits with online data augmentation applied exclusively during training.
- **Robust Model Diversity:** Evaluates three primary models targeting distinct operational priorities: high feature accuracy (ConvNeXt-Tiny), balanced scaling (EfficientNetV2-S), and lightweight edge deployment (MobileNetV2).
- **High Resolution Processing:** Images processed at $384 \times 384$ resolution normalized with ImageNet statistics to capture granular texture details.
- **Cross-Validation Rigor:** 5-Fold Stratified Cross-Validation on the training/validation partitions before single-pass evaluation on the locked test set.

---

## 📊 Dataset & Visualizations

The project snapshot comprises **12,259 JPEG images** categorized into **10 distinct classes**:

| Class | Image Count | Class | Image Count |
| :--- | :--- | :--- | :--- |
| **Clothes** | 1,892 | **Cardboard** | 1,411 |
| **Glass** | 1,736 | **Paper** | 1,336 |
| **Plastic** | 1,597 | **Metal** | 930 |
| **Shoes** | 1,449 | **Battery** | 756 |
| **Biological** | 699 | **Trash** | 453 |

### Dataset Samples
<img width="800" height="600" alt="trash_72" src="https://github.com/user-attachments/assets/ccb6b6cb-be62-45cd-b8c3-78c045f68cd2" />
Dataset Class Samples
<img width="283" height="264" alt="glass_355" src="https://github.com/user-attachments/assets/4641e2d9-ca53-45ce-8764-be20afb8cf20" />
<img width="800" height="600" alt="plastic_272" src="https://github.com/user-attachments/assets/2e0ed5d4-5a35-4627-981f-62df76353031" />
<img width="800" height="600" alt="paper_480" src="https://github.com/user-attachments/assets/eb64a928-1c05-452a-897e-5c25b8af044b" />
<img width="262" height="193" alt="biological_577" src="https://github.com/user-attachments/assets/7a5095f8-b24c-4561-8c77-6260d6f49bff" />


### Dataset Partitioning & Class Distribution
<img width="1800" height="1100" alt="split_ratio_comparison" src="https://github.com/user-attachments/assets/fd7a72e8-30e7-40cb-83b8-4d8dd04888fc" />


- **Training Set:** 9,194 images (Balanced)
- **Validation Set:** 1,839 images
- **Held-Out Test Set (Locked):** 1,226 images

---

## ⚙️ Model Architecture & Pipeline

Pipeline Diagram
<img width="1306" height="816" alt="Gemini_Generated_Image_5900lr5900lr5900" src="https://github.com/user-attachments/assets/28a3afbc-8532-4699-bb2c-517d662a0189" />

