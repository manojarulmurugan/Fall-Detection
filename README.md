# 🎥 Fall Detection using ConvLSTM & LRCN

## 📌 Project Overview
This project implements **fall detection** using deep learning with **ConvLSTM and LRCN models**. The models are trained on **Harvard Dataverse's Fall Vision Dataset** to recognize fall events in videos.

## **🚀 Version 2.0 Updates**
- Implemented **ConvLSTM2D** for spatiotemporal learning.
- Improved **LRCN (Long-term Recurrent Convolutional Networks)** model.
- Addressed overfitting issues from **Version 1.0** by:
  - **Data Augmentation**
  - **Batch Size Adjustment**
  - **Learning Rate Tuning**
  - **Sequence Length Optimization** 

## **🗂️ Dataset**
- **Source**: Harvard Dataverse's Fall Vision Dataset ([Link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK))
- **Data Type**: Video sequences labeled as `fall` and `no_fall`.
- **Preprocessing**:
  - Extracted frames from videos.
  - Normalized pixel values to `[0,1]` range.
  - Converted sequence data into **time-series tensors**.

---

## **🛠️ Model Architectures**
### **1️⃣ ConvLSTM2D Model**
- A **hybrid CNN-LSTM** approach that captures both **spatial and temporal** features.
- Used **ConvLSTM2D layers** to process sequential video frames.

## Timeline:
<img width="892" alt="Screenshot 2024-12-11 at 2 39 51 AM" src="https://github.com/user-attachments/assets/7f4be7a4-df78-4604-b8fe-c3cae78c7871">

📌 **Findings from ConvLSTM V1.0**:
- **Overfitting observed** (Training accuracy: 95%, Validation accuracy: ~75%).
- **Validation loss fluctuated**, suggesting need for **regularization**.
![image](https://github.com/user-attachments/assets/c8843f1d-190d-4ef3-8be0-c1853b317311)

✅ **Improvements in V2.0**:
- Increased **sequence length** (from 3 → 10 frames).
- Added **Dropout (0.3) and L2 Regularization**.
- Optimized **learning rate** with adaptive schedulers.
![image](https://github.com/user-attachments/assets/1acc6872-08b0-4030-9294-d35b44938ce1)

### **2️⃣ LRCN Model**
- **Combines CNN feature extraction with LSTMs** to model fall events over time.

📌 **Findings from LRCN V1.0**:
- Poor learning capacity (`Train Accuracy: 55%`, `Validation Accuracy: 60%`).
- Model struggled to learn meaningful **temporal patterns**.
![image](https://github.com/user-attachments/assets/6f772837-8156-405d-b89d-82d61c7be6df)

✅ **Improvements in V2.0**:
- Increased **LSTM hidden units** (from 32 → 64).
- Adjusted batch size (from 4 → 8).
- Reduced learning rate (`0.0001` for Adam Optimizer).
![image](https://github.com/user-attachments/assets/bea3b1f6-af79-4999-aa40-bcee3bcb864c)

---

## 📈 Key Results

| Model | Training Accuracy | Validation Accuracy |
|--------|----------------|------------------|
| **ConvLSTM V2.0** | **91.5%** | **82.0%** |
| **LRCN V2.0** | **88.3%** | **79.5%** |

📌 **Key Insights**:
- Increasing **sequence length** (frames per video) improved results.
- ConvLSTM **outperforms LRCN**, indicating **better spatial-temporal feature extraction**.
- Further tuning is needed to **reduce validation loss fluctuations**.

---

## 📈 Visualizations

## 🎥 Sample Output Video
🔗 [Click here to watch]([https://drive.google.com/file/d/FILE_ID/view](https://drive.google.com/file/d/1wj3FqGj4hPY0v7jr4Bh4YC07gxucsrwc/view?usp=sharing))

### **Confusion Matrix**
![Confusion Matrix](reports/images/confusion_matrix.png)

---

## **🔍 Next Steps**
1. **Predict in Real Time**:
   - Make the model predict in **real-time** and **Host the Application**.
2. **Further Regularization**:
   - Test different **dropout rates** and **batch normalization**.
3. **Frame Selection Strategies**:
   - Instead of **uniform sampling**, explore **randomized sampling** for better representation.
4. **Alternative Architectures**:
   - Try **Transformer-based models** for video classification.

---
