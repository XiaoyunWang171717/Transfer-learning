# MLP Transfer Learning for Adduct Prediction

This project implements a transfer learning approach using Multi-Layer Perceptron (MLP) for predicting adduct properties in mass spectrometry data.

## Features

- Transfer learning implementation using PyTorch
- Molecular fingerprint generation using RDKit
- Model performance evaluation with multiple metrics (R², RMSE, MAE)
- Support for both direct prediction and transfer learning approaches

## Requirements

- Python 3.7+
- PyTorch
- RDKit
- scikit-learn
- pandas
- numpy

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in CSV format with 'SMILES' and 'logNAH' columns
2. Place your pre-trained model in the specified directory
3. Run the transfer learning script:
```bash
python transfer_learning_and_prediction.py
```

## Project Structure

- `transfer_learning_and_prediction.py`: Main script for transfer learning and prediction
- `requirements.txt`: List of required Python packages
- `README.md`: Project documentation

## Model Architecture

The model uses a Multi-Layer Perceptron (MLP) with the following features:
- Input layer: 2048 nodes (molecular fingerprint)
- Hidden layers: Configurable sizes
- Output layer: 1 node (prediction of logNAH)

## Performance Metrics

The model evaluation includes:
- R² Score
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)

## License

[Your chosen license] 