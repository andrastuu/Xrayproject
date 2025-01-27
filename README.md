# Using Deep Learning to Perform Multi-Class Classification on the COVID-19 Chest X-ray Dataset

---

## **1. Problem Definition**

Computer vision is crucial in medical data analysis as it provides:
- **Accurate Diagnoses:** Automated identification of abnormalities.
- **Efficiency:** Reduced time and cost of manual processes.
- **Trend Identification:** Pattern recognition in large datasets, aiding prevention and treatment decisions.

For example, MediPredict uses deep learning to predict disease risks based on medical data ([MediPredict](https://medipredict.com)).

---

## **2. Dataset Preprocessing**

### Dataset Overview:
- **Total Images:** 6,392
- **Classes:**
  - Bacterial Pneumonia: 2,816
  - COVID-19: 127
  - Healthy: 1,606
  - Viral Pneumonia: 1,843
- **Image Dimensions:** 156 x 156 pixels

### Preprocessing Steps:
1. **One-Hot Encoding:** Efficient representation of categorical data.
2. **Data Splitting:** Stratified splits:
   - **Training Set:** 4,090 images
   - **Validation Set:** 1,023 images
   - **Test Set:** 1,279 images
3. **Normalization:** Reduces noise and improves model accuracy.

---

## **3. Baseline Model**

### Architecture:
The baseline model consists of:
- 4 Convolutional layers with ReLU activation.
- Max Pooling layers to reduce dimensionality.
- 2 Dense layers (32 neurons each).
- A Softmax output layer for classification.



### Performance:
- **Validation Accuracy:** 78.1%
- **Test Accuracy:** 76.4%

<div align="center">
  <img src="images/baseline%20metrics.png" alt="Baseline Metrics" width="48%">
</div>

### Metrics and Confusion Matrices:
<div align="center">
  <img src="images/baseline%20matrix%20val.png" alt="Baseline Validation Confusion Matrix" width="48%">
  <img src="images/baseline%20matrix%20test.png" alt="Baseline Test Confusion Matrix" width="48%">
</div>


---

## **4. Improved (Fine-Tuned) Model**

### Improvements:
1. **Data Augmentation:** 
   - Rotation (0.3), zooming (0.1), and vertical flipping.
2. **Architecture Enhancements:**
   - Increased convolutional layers (6 total).
   - Dropout layers (rate: 0.2) and L2 regularization.
   - Early stopping to prevent overfitting.
3. **Hyperparameter Tuning:** 
   - Batch size: 32
   - Learning rate: 0.0001

### Performance:
- **Validation Accuracy:** 81.5%
- **Test Accuracy:** 81.9%

<div align="center">
  <img src="images/tuned%20metrics.png" alt="Tuned Metrics" width="48%">
</div>

### Metrics and Confusion Matrices:
<div align="center">

  <img src="images/tuned%20matrix%20val.png" alt="Tuned Validation Confusion Matrix" width="48%">
    <img src="images/tuned%20matrix%20test.png" alt="Tuned Test Confusion Matrix" width="48%">
</div>

<div align="center">

</div>


---

## **5. Transfer Learning Model**

### Approach:
- Utilized **VGG16** and **ResNet50** architectures.
- VGG16 performed better after modifications:
  - Dropout layers: 0.2
  - L2 regularization
  - Adam optimizer with a learning rate of 0.01.
- Similar augmentation techniques were applied.

### Performance:
- Results were comparable to the fine-tuned model.

---

## **6. Discussion**

### Key Findings:
- The models achieved reasonable accuracy despite data limitations.
- Augmentation and fine-tuning helped reduce overfitting.
- COVID-19 classification accuracy remained low due to class imbalance.

### Limitations:
- Computational resources limited hyperparameter tuning.
- Additional experiments with Keras Tuner or other optimizers could improve results.

---

## **7. References**

1. Ahmed F., Bukhari S.A.C., Keshtkar F. (2021). *A deep learning approach for COVID-19 viral pneumonia screening with X-Ray images*. DOI: 10.1145/3431804.
2. Meedeniya D. et al. (2022). *Chest X-ray analysis empowered with deep learning: A systematic review*. DOI: 10.1016/j.asoc.2022.109319.
