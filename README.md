# Transformer-Based Next Token Prediction

## Description
This repository implements a transformer model in PyTorch for the task of predicting the next token in a sequence. The model utilizes a custom-built transformer layer with multi-head self-attention, embedding, and positional encoding to process sequences of characters from Shakespeare's plays. The project includes a mini-batch data sampler for training, the implementation of the transformer architecture, and the use of a next-token prediction loss function. Additionally, the repository explores the effects of positional encoding on model performance, with detailed hyperparameter tuning, validation loss comparison, and attention weight analysis.

Ideal for learning about the inner workings of transformers and experimenting with language modeling tasks, this project also includes practical insights into the tradeoffs between using and not using positional encodings.

## Features:
- Transformer model built from scratch using PyTorch
- Implementation of positional encoding and self-attention mechanisms
- Next-token prediction for character-level language modeling
- Hyperparameter tuning and performance comparison with and without positional encoding
- Attention weight analysis to understand model behavior
