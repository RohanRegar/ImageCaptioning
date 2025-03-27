# Image Captioning with Deep Learning
This project is part of the Deep Learning (CSL4020) course at IITJ.
## Objective
This project implements an image captioning system using deep learning techniques. The model generates descriptive captions for images by leveraging a combination of a Convolutional Neural Network (CNN) for feature extraction and a Recurrent Neural Network (RNN) for sequence generation. The goal is to compare different decoder architectures, including:

- **LSTM-based decoder**
- **GRU-based decoder**
- **Attention-based LSTM decoder**

The effectiveness of these architectures is evaluated using BLEU scores and other performance metrics.

## Dataset
This project uses the **MSCOCO 2017 dataset**, a large-scale dataset for image captioning tasks. The dataset consists of:

- **Images**: High-quality images covering a wide range of objects and scenes.
- **Captions**: Each image is paired with multiple human-annotated captions that describe the content of the image.
- **Annotations**: JSON files contain structured caption data for training and evaluation.

### Dataset Structure
- **Training set**: Used for training the captioning models.
- **Validation set**: Used for performance evaluation.
- **Test set**: (Optional) Used for final evaluation if required.

The dataset is preprocessed by tokenizing captions, building a vocabulary, and transforming images into feature vectors using a pretrained **ResNet-50** model.

---
Will further add details about the **Model Architectures, Results, and Evaluation** in later sections.
