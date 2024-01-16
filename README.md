## Simple Transformer Implementation
This repository contains a simplified (single block) implementation of the transformer model both for encoder-only model and encoder-decoder model.

* The encoder-only model is trained on a toy dataset of spam detection.
* The encoder-decoder model is trained on a toy string reversing task.

#### Run the encoder-only training:
`python simple_transformer/train.py `

#### Run the encoder-decoder model training:
`python simple_encoder_decoder/train_seq2seq.py `