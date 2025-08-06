# 💵 Currency Note Recognition using Transfer Learning

This project classifies Indian currency notes (₹100, ₹200, ₹500, ₹2000) using **MobileNetV2** with **transfer learning**. The pipeline includes dataset preparation, image preprocessing, data augmentation, model training, evaluation, and prediction on external images.

---

## 📁 Dataset Structure

The dataset must be organized as follows:

```
/dataset
  ├── train/
  │   ├── 100/
  │   ├── 200/
  │   ├── 500/
  │   └── 2000/
  ├── val/
  └── test/
```

Each subfolder should contain `.jpg` images for that denomination.

You can use the provided split script to divide a folder of images into train/val/test:

```python
# Automatically splits data into train/val/test
# using sklearn's train_test_split
```

---

## 📦 Requirements

Install the following Python packages:

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn
```

---

##  Model Architecture

- **Backbone:** MobileNetV2 (pretrained on ImageNet)
- **Classifier Head:**
  - GlobalAveragePooling2D
  - Dense(128) + Dropout(0.3)
  - Dense(32) + Dropout(0.2)
  - Dense(4) + Softmax

---

##  Data Pipeline

- Images resized to **512x512**
- RGB conversion and normalization (0-1)
- One-hot encoding on labels
- Data augmentation using:
  - Rotation: ±30°
  - Zoom: up to 20%
  - Horizontal flips

---

##  Model Training

- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Epochs:** 10 (customizable)
- **Batch Size:** 8

```python
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)
```

---

##  Evaluation Metrics

- **Accuracy:** 99.32% on test set
- **Confusion Matrix** and **Classification Report** included
- Visual plots for:
  - Accuracy trends
  - Confusion Matrix
  - Sample predictions

---

##  Example Results

```
Classification Report:
              precision    recall  f1-score   support
         100       0.98      1.00      0.99        46
         200       1.00      0.97      0.98        31
        2000       1.00      1.00      1.00        36
         500       1.00      1.00      1.00        34
```

---

##  Visual Output

Sample prediction output:

- Correct predictions shown in **blue**
- Incorrect predictions shown in **red**

```python
# Shows 32 random predictions with color-coded results
```

---

## 💾 Save & Load Model

```python
model.save('/path/to/currency_note_model.h5')
model = load_model('/path/to/currency_note_model.h5')
```

---

##  Predict on New Image

```python
predict_single_image('/path/to/note.jpg')
```

Returns:
```
Predicted class: 200
```

---

## 📌 To-Do

- [ ] Add more denominations or currencies
- [ ] Fine-tune MobileNetV2 (currently frozen)
- [ ] Optimize input shape for better inference speed

---


