# Image Captioning with CNN-RNN Architecture

## Project Overview

This project implements an image captioning system using a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). The system is designed to automatically generate descriptive captions for images, specifically trained on the UCM (UC Merced Land Use) dataset.

## Project Structure

```
Final_folder/
├── CNN_training.ipynb          # CNN-based image captioning training
├── RNN_training.ipynb          # RNN-based image captioning training  
├── new_captions.txt            # Processed caption dataset
├── UCM_captions/               # Main dataset directory
│   ├── dataset.json            # Dataset metadata and annotations
│   └── Features/               # Pre-extracted image features
│       ├── test_features.pkl   # Test set features
│       ├── train_features.pkl  # Training set features
│       └── val_features.pkl    # Validation set features
└── README.md                   # This file
```

## Dataset

The project uses the **UCM (UC Merced Land Use) dataset**, which contains aerial images of different land use categories. The dataset includes:

- **Images**: High-resolution aerial photographs
- **Captions**: Multiple descriptive captions per image
- **Features**: Pre-extracted CNN features for efficient training
- **Split**: Training (70%), Validation (15%), and Test (15%) sets

## Architecture

### CNN Component
- **Purpose**: Extracts visual features from images
- **Implementation**: Uses pre-trained CNN models to generate feature vectors
- **Output**: Fixed-length feature representations of images

### RNN Component  
- **Purpose**: Generates natural language captions from visual features
- **Architecture**: LSTM-based sequence model with attention mechanisms
- **Training**: Uses teacher forcing with start/end sequence tokens

## Key Features

- **Multi-modal Learning**: Combines computer vision and natural language processing
- **Attention Mechanism**: Focuses on relevant image regions during caption generation
- **BLEU Score & CIDER Evaluation**: Measures caption quality using standard NLP metrics (BLEU for n-gram overlap, CIDER for semantic similarity)
- **Data Augmentation**: Multiple captions per image for robust training
- **Feature Extraction**: Pre-computed features for faster training iterations

## Requirements

### Core Dependencies
- TensorFlow/Keras
- NumPy
- Pandas
- NLTK
- scikit-learn
- Pickle

### Optional Dependencies
- Google Colab (for cloud-based training)
- GPU support (recommended for faster training)

## Usage

### 1. Data Preparation
The project includes utilities for:
- Dataset splitting and organization
- Feature extraction and storage
- Caption preprocessing and tokenization

### 2. Training
Two main training approaches are provided:

#### CNN Training (`CNN_training.ipynb`)
- Focuses on visual feature extraction
- Includes data splitting utilities
- Prepares features for caption generation

#### RNN Training (`RNN_training.ipynb`)
- Implements the caption generation model
- Uses extracted features as input
- Trains the language model end-to-end

### 3. Evaluation
- **BLEU Score**: Measures n-gram overlap with reference captions
- **CIDER Score**: Evaluates semantic similarity and consensus
- **Perplexity**: Evaluates model confidence in predictions
- **Human Evaluation**: Qualitative assessment of generated captions

## Training Process

1. **Data Loading**: Load pre-extracted features and captions
2. **Tokenization**: Convert text to numerical sequences
3. **Model Building**: Construct CNN-RNN architecture
4. **Training**: Train with teacher forcing and validation
5. **Evaluation**: Assess performance on test set
6. **Inference**: Generate captions for new images

## Model Architecture Details

### Input Processing
- Images are processed through pre-trained CNN
- Features are flattened and normalized
- Captions are tokenized and padded

### Network Layers
- **Embedding Layer**: Converts tokens to dense vectors
- **LSTM Layers**: Process sequential caption information
- **Dense Layers**: Generate vocabulary predictions
- **Dropout**: Regularization to prevent overfitting

### Training Strategy
- **Teacher Forcing**: Uses ground truth captions during training
- **Beam Search**: Generates captions during inference
- **Early Stopping**: Prevents overfitting on validation set

## Performance Metrics

- **BLEU Score**: N-gram overlap with reference captions
- **CIDER Score**: Semantic similarity and consensus evaluation


