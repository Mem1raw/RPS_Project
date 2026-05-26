# ✊✋✌️ Rock-Paper-Scissors Image Classifier

A Convolutional Neural Network (CNN) built with TensorFlow/Keras that classifies hand gestures into **Rock**, **Paper**, or **Scissors** from images.

---

## 📌 Project Overview

This project uses deep learning to recognize Rock-Paper-Scissors hand gestures from image data. The model is trained on a publicly available dataset and achieves strong classification performance using a custom CNN architecture with data augmentation.

---

## 🗂️ Dataset

The dataset used in this project is the [Rock-Paper-Scissors Dataset](https://www.kaggle.com/), which contains labeled images of hand gestures split into three directories:

```
RPS_Data/
└── Rock-Paper-Scissors/
    ├── train/
    ├── validation/
    └── test/
```

Each directory contains subdirectories for each class: `rock`, `paper`, and `scissors`.

---

## 🧠 Model Architecture

The model is a Sequential CNN with the following structure:

| Layer | Details |
|---|---|
| Rescaling | Normalizes pixel values to [0, 1] |
| Conv2D + MaxPooling | 32 filters, 3×3 kernel |
| Conv2D + MaxPooling | 64 filters, 3×3 kernel |
| Conv2D + MaxPooling | 128 filters, 3×3 kernel |
| Conv2D + MaxPooling | 128 filters, 3×3 kernel |
| Flatten | Retains spatial features for hand shape recognition |
| Dense | 512 units, ReLU activation |
| Dropout | 0.5 rate to prevent overfitting |
| Dense (Output) | 3 units, Softmax activation |

---

## 🔄 Data Augmentation

To improve generalization, the following augmentations are applied to training images:

- Random horizontal and vertical flipping
- Random rotation (±20%)
- Random zoom (±20%)

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Image Size | 64 × 64 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Max Epochs | 20 |
| Early Stopping | Patience = 5 (monitors val_loss) |
| Random Seed | 20 |

---

## 📊 Evaluation

After training, the model is evaluated on the test set. The following metrics are reported:

- Overall test accuracy and loss
- Per-class accuracy
- Confusion matrix
- Full classification report (precision, recall, F1-score)

Training and validation accuracy/loss curves are also plotted alongside the test accuracy baseline.

---

## 🚀 Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

### Running the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/RPS_Project.git
cd RPS_Project
```

2. Place your dataset in the following structure under the project root:

```
RPS_Data/Rock-Paper-Scissors/train/
RPS_Data/Rock-Paper-Scissors/validation/
RPS_Data/Rock-Paper-Scissors/test/
```

3. Update the dataset paths in `main.py` if necessary, then run:

```bash
python main.py
```

---

## 📁 Project Structure

```
RPS_Project/
├── RPS_Data/          # Dataset directory (not included in repo)
├── .venv/             # Virtual environment (not included in repo)
├── main.py            # Main training and evaluation script
└── README.md
```

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy
- scikit-learn
- Matplotlib

---
