# Age Detection using ResNet50 and UTK-Face-Revised

This project implements a Deep Learning pipeline to predict the biological age of individuals from facial images. It leverages a revised version of the UTK-Face dataset and employs transfer learning with a ResNet50 backbone.

## 🚀 Project Overview
The goal of this project is to perform age regression (predicting the exact age) and age group classification using computer vision. The model is trained on a diverse dataset of facial images to ensure robustness across different genders and races.

## 📊 Dataset: UTK-Face-Revised
The project uses the [deedax/UTK-Face-Revised](https://huggingface.co/datasets/deedax/UTK-Face-Revised) dataset hosted on Hugging Face.
- **Total Images:** ~8,500 samples.
- **Features:**
  - `image`: 200x200 pixel facial images.
  - `age`: Continuous numerical value.
  - `gender`: Male/Female labels.
  - `race`: White, Black, Asian, Indian, and Others.
  - `age_group`: Categorical groups (e.g., Baby, Child, Young adult, etc.).

## 🛠️ Technical Stack
- **Framework:** TensorFlow / Keras.
- **Architecture:** ResNet50 (Pre-trained on ImageNet).
- **Data Handling:** `datasets` library (Hugging Face API), Pandas, NumPy.
- **Visualization:** Matplotlib, Seaborn.
- **Image Processing:** OpenCV (cv2), PIL.

## 🏗️ Model Architecture
The model utilizes a **Transfer Learning** approach:
1. **Base Model:** ResNet50 backbone with pre-trained weights.
2. **Preprocessing:** Input images are resized and passed through the `resnet50.preprocess_input` function.
3. **Head:** Custom Dense layers added on top of the ResNet base for the specific regression/classification task.
4. **Optimization:**
   - `EarlyStopping`: To prevent overfitting by monitoring validation loss.
   - `ReduceLROnPlateau`: To dynamically adjust the learning rate during training.

## 📈 Exploratory Data Analysis (EDA)
The notebook includes detailed analysis of the dataset distribution:
- **Age Distribution:** Histograms showing the frequency of different age groups.
- **Demographic Balance:** Bar charts visualizing the distribution of Gender and Race within the training set.


## ⚙️ Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/age-detection-resnet50.git
   ```
2. **Install dependencies:**
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn opencv-python datasets pillow
   ```
3. **Run the Notebook:**
   Open `Age_Detection.ipynb` in Jupyter or Google Colab to train and evaluate the model.

## 📝 Results
The model's performance is tracked using standard metrics for regression (like MAE) and classification (Accuracy), with training logs and loss curves provided within the notebook.
