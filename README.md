# EEWnet - Earthquake Early Warning Network

## Project Overview
EEWnet is a real-time seismic response prediction method of high-rise buildings based on deep learning for earthquake early warning

Details of the method can be referred to: Real-time seismic response prediction method of high-rise buildings based on deep learning for earthquake early warning

## Environment Requirements
- Python 3.9.13
- CUDA 12.4

### Dependencies
torch==2.4.1+cu124
numpy==1.22.4
pandas==2.2.3
scikit-learn==1.5.2
matplotlib==3.7.2

## Dataset Description
- Input file: input.csv
- Data preprocessing:
  - Logarithmic transformation of data
  - Removal of outliers and NaN values
  - Features include:
    - nStory
    - storyheight
    - year
    - strutype
    - Earthquake Magnitude
    - EpiD
    - Vs30
    - Peak_Ground_Velocity
    - Significant_Duration
    - Arias_Intensity
    - Cumulative_Absolute_Velocity
    - Peak_Ground_Velocity

## Model Architecture
- Input layer: 12 feature nodes
- Hidden layers:
  - 512 nodes × 4 layers
  - 256 nodes × 2 layers
  - 128 nodes × 1 layer
  - 64 nodes × 1 layer
  - 32 nodes × 1 layer
- Output layer: 1 node
- Activation function: LeakyReLU(0.01)
- Regularization: BatchNormalization and Dropout

## Training Parameters
- Learning rate: 0.0005
- Batch size: 512
- Epochs: 300
- Optimizer: Adam
- Loss function: MSE

## Model Evaluation
The model is evaluated using the following metrics:
- R² score: Measures the goodness of fit between predicted and actual values
- Mean Absolute Error (MAE): Average absolute difference between predicted and actual values
- Mean Squared Error (MSE): Average of squared differences between predicted and actual values
- Root Mean Square Error (RMSE): Square root of MSE, representing the standard deviation of predictions

## Output Files
The model training process generates the following files:
- loss_{timestamp}.png: Loss curve during training
- Predicted_vs_actual_{timestamp}.png: Scatter plot comparing predicted vs actual values
- my_best_model.pth: Saved best model weights
- true_pre.txt: Model prediction results
- metrics.txt: Contains all evaluation metric results
- loss.txt: Records loss values during training and testing

## Usage Instructions
1. Data Preparation:
   - Place training data in the specified directory
   - Ensure data format meets requirements

2. Model Training:
   ```bash
   python EEWnet.py
   ```

3. View Results:
   - Check generated charts and metric files in the output directory
   - Use generated my_best_model.pth for predictions
