
# 100 Sports Image Classification

This repository contains code for classifying 100 different sports categories using multiple CNN architectures. The models are trained on the "Sports Classification" dataset available on Kaggle. The purpose of this project is to evaluate the performance of various CNN architectures and determine the model best suited for this classification task.

## Project Overview

Sports classification involves distinguishing images from different sports categories. This project explores several convolutional neural network (CNN) architectures to classify images into 100 distinct sports categories. Each model’s performance is compared to select the best-performing architecture for this dataset.

## Dataset

**Dataset Link:** [Sports Classification Dataset on Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification)

The dataset contains images of 100 different sports categories, suitable for multi-class image classification tasks.

### Dataset Details:
- **Number of Classes:** 100
- **Number of Images:** ~10,000
- **Image Resolution:** Variable

The dataset includes a variety of sports, making it an ideal choice for training a robust classifier.

## Models and Architectures

The following CNN architectures were trained and evaluated in this project:
1. **AlexNet**
2. **VGG16**
3. **VGG19**
4. **GoogLeNet**

Each model is trained and fine-tuned on the dataset, and the performance metrics (accuracy and loss) are recorded to compare results.

## Training Details

### Libraries Used:
- **PyTorch**: For model building and training
- **Torchvision**: For pre-trained models and data transformations
- **NumPy and Pandas**: For data manipulation
- **Matplotlib**: For visualizing results

### Hyperparameters:
- **Epochs**: 15
- **Batch Size**: 32
- **Learning Rate**: Adjusted per model
- **Optimizer**: SGD/Adam with momentum (varied across experiments)
- **Loss Function**: Cross-Entropy Loss

### Data Augmentation:
- Random cropping, flipping, and resizing techniques were used to augment the dataset and improve model generalization.

## Evaluation

Each model was evaluated using:
- **Training Accuracy and Loss**
- **Validation Accuracy and Loss**

The model with the highest validation accuracy and lowest validation loss is recommended as the best classifier for this dataset.

## Results

| Model       | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------------|--------------------|---------------------|---------------|-----------------|
| AlexNet     | ...                | ...                 | ...           | ...             |
| VGG16       | ...                | ...                 | ...           | ...             |
| VGG19       | ...                | ...                 | ...           | ...             |
| GoogLeNet   | ...                | ...                 | ...           | ...             |

*Note:* Fill in the table above with the actual results obtained after training.

## Getting Started

### Prerequisites
- **Python 3.7+**
- **PyTorch**
- **Torchvision**
- **NumPy, Pandas, Matplotlib**

### Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/yourusername/100-sports-image-classification.git
cd 100-sports-image-classification
pip install -r requirements.txt
```

### Running the Code
1. Download the dataset from the [dataset link](https://www.kaggle.com/datasets/gpiosenka/sports-classification) and place it in the project directory.
2. Run the training script for each model or open the Kaggle notebook for an interactive version:
   - **Kaggle Notebook**: [100 Sports Image Classification Notebook](https://www.kaggle.com/code/asad1212/100-sports-image-classification/edit)

## Conclusion

This project provides insights into the effectiveness of different CNN architectures for sports classification. Based on the results, the optimal model can be selected and further fine-tuned for real-world applications.
