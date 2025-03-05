# VolGAN: A Deep Learning Approach to Volatility Modeling

## Overview

VolGAN is a deep learning framework for modeling and predicting financial market volatility using Generative Adversarial Networks (GANs). By leveraging the power of adversarial training, VolGAN aims to capture complex market dynamics and provide more robust volatility estimations compared to traditional models.

## Features

- Implements a GAN-based approach for financial volatility modeling
- Supports training on historical financial data
- Provides volatility forecasts and synthetic volatility path generation
- Extensible architecture for experimentation with different network structures

## Installation

To use VolGAN, clone this repository and install the required dependencies:

```bash
git clone https://github.com/ekopen/VolGAN.git
cd VolGAN
pip install -r requirements.txt
```

## Usage

To see relevant code for data preprocessing, model training, model usage and model evaluation, please refer to the `main.ipynb` file in the repository.

The model itself and relevant functions can be found in any of the relevant `.py` files:
- `VolGANSwaps.py` abstracts most of the processing, training and usage functions
- `Inputs.py` provides functions to interact with the originally provided dataset
- `Prices and Greeks.py` provides functions to price the instrument given volatilities
- To be continued. 
