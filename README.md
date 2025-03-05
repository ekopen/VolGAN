# VolGAN: A Deep Learning Approach to Volatility Modeling

## Overview

VolGAN is a deep learning framework for modeling and predicting financial market volatility using Generative Adversarial Networks (GANs). By leveraging the power of adversarial training, VolGAN aims to capture complex market dynamics and provide more robust volatility estimations compared to traditional models.

## Features

- Implements a GAN-based approach for financial volatility modeling
- Supports training on historical financial data
- Provides volatility forecasts and synthetic volatility path generation
- Extensible architecture for experimentation with varying hyperparameter selection

## Installation

To use VolGAN, clone this repository and install the required dependencies:

```bash
git clone https://github.com/ekopen/VolGAN.git
cd VolGAN
pip install -r requirements.txt
```

## Usage

To see relevant code for data preprocessing, model training, model usage and model evaluation, please refer to the `main.ipynb` file in the repository.

The model itself and relevant functions can be found in `VolGANSwaps.py` which abstracts most of the processing, training and usage functions.

Other relevant files include the following:
- `Inputs.py` provides functions to interact with the originally provided dataset
- `Prices and Greeks.py` provides functions to price the instrument given implied volatilities
  - `treasury_cmds.py` are helper functions used in pricing
- `reweighting.py` and `reweighting_main.ipynb` provide implementation of the scenario reweighting process provided in the paper
- `sofr_libor_conversion.ipynb` provides methodology for constructing synthetic SOFR data from pre-LIBOR-to-SOFR transtion
- `Evaluation.py` provides additional visualization functions and methods to compare generated surfaces and true surfaces
