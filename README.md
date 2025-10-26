# Waste Classifier Model

A simple deep learning model that classifies images of waste into one of four categories — **General Waste**, **Hazardous**, **Organic**, or **Recyclable** — helping automate waste segregation and recycling processes.

---

## Overview
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow and Keras**.
The model learns to identify waste categories from images using data augmentation and normalization techniques.

It can be integrated into applications for **smart waste bins**, **waste management systems**, or **environmental monitoring tools**.

---

## Dataset
The dataset(not made publicly available) consists of images collected manually, organized into four classes:

```
Waste_Segregation_DatasetV2/
│
├── General/
├── Hazardous/
├── Organic/
└── Recyclable/
```

Each subfolder contains representative images of the corresponding waste type.
If you wish to use this notebook with your own dataset, follow the same folder structure.

---

## Model Architecture

```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
```
---
## Training Configuration
- **Framework:** TensorFlow / Keras
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Batch size:** 64
- **Image size:** 224 × 224
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 🧪 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Waste-classifier-model.git
   cd Waste-classifier-model
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow matplotlib numpy
   ```

3. Open the notebook:
   ```bash
   jupyter notebook WS5.ipynb
   ```
   or run it directly on **Google Colab**.

4. Update the `dataset_path` variable to the location of your dataset.

---

## License
This project is licensed under the [MIT License](LICENSE).
