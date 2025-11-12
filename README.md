# Brain Tumor Detection Using Computer Vision

This project applies deep learning and computer vision techniques to automatically detect and classify brain tumors from MRI images.
A pre-trained VGG16 convolutional neural network is fine-tuned using transfer learning to distinguish between Glioma, Meningioma, Pituitary, and No Tumor MRI scans.

## Project Overview

Brain tumor detection using MRI images is a critical challenge in medical imaging and diagnostics. Traditional manual diagnosis is time-consuming and prone to human error.
This project leverages transfer learning with VGG16 to automate classification, achieving high accuracy with limited training data.

Implemented in Google Colab, the system includes data preprocessing, image augmentation, model training, and performance evaluation.

## Dataset Structure

The dataset is divided into Training and Testing folders, each containing four tumor classes:

dataset/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/


Each subfolder includes MRI images specific to that tumor category.

## Image Preprocessing and Augmentation

To enhance model robustness and reduce overfitting, custom augmentation techniques were applied:

def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    image = np.array(image) / 255.0  # Normalize pixel values
    return image


Key preprocessing steps:

Random brightness and contrast variations

Normalization of pixel values to [0, 1]

Label encoding for categorical class names

Batch generation via a custom datagen() generator for efficient training

## Model Architecture (Transfer Learning with VGG16)

The model is built upon VGG16 (pre-trained on ImageNet), with the top layers removed and custom layers added for classification.

**Architecture Summary**

Input Shape: (128, 128, 3)

Base Model: VGG16 (include_top=False, weights='imagenet')

Frozen Layers: All except the last three convolutional layers

**Added Layers:**

Flatten

Dropout(0.3)

Dense(128, ReLU)

Dropout(0.2)

Dense(4, Softmax)

**Model Compilation**
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

**Training Parameters**
batch_size = 20
epochs = 5
steps = len(train_paths) / batch_size


The model is trained using the custom data generator, which dynamically augments and batches data.

## Model Training and Evaluation

**During training:**

Data augmentation improves generalization

Dropout reduces overfitting

Accuracy and loss curves are plotted for training progress visualization

After training, the model demonstrates strong classification accuracy across all four tumor types on the test dataset.

(You can insert your final accuracy or confusion matrix here.)

## Technologies Used

Python 3.x

TensorFlow / Keras

NumPy, Pandas

OpenCV

Matplotlib, Seaborn

PIL (ImageEnhance)

Google Colab

## Project Structure
├── brain_tumor_detection_using_computer_vision.ipynb
├── dataset/
│   ├── Training/
│   └── Testing/
├── README.md
└── requirements.txt

## How to Run

Clone the repository:

git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection


Open in Google Colab or Jupyter Notebook:

brain_tumor_detection_using_computer_vision.ipynb


Mount Google Drive and update dataset path.

Run all cells sequentially to:

Preprocess data

Train model

Evaluate accuracy and visualize results

## Future Enhancements

Implement Grad-CAM for visual interpretability

Use advanced CNN architectures like EfficientNet or ResNet50

Extend to multi-modal MRI datasets (T1, T2, FLAIR)

Deploy as a web app using Streamlit or Flask

