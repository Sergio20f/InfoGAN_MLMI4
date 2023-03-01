# InfoGAN_MLMI4

This repository contains the PyTorch implementation of InfoGAN, a generative adversarial network that can learn interpretable representations in an unsupervised manner. InfoGAN extends the traditional GAN by introducing additional latent codes that capture specific attributes of the generated images.

## Requirements
* Python 3.10.9
* PyTorch 1.13.1
* torchvision 0.13.1
* matplotlib 3.6.2
* tensorboardX 2.2

## Usage
1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt` (TBC)
3. Run `train.py` to start training the model. Change the training parameters in the file according to your preference.
4. To change the network architecture, modify `GAN.py` file.

## Files
* `train.py` - main script for training the InfoGAN model
* `GAN.py` - contains the network architectures for the discriminator, generator, and regularization network
* `utils.py` - contains utility functions for sampling noise and calculating log-likelihoods
* `InfoGAN.ipynb` - Jupyter notebook for testing the model

## Training process
The training process consists of training the Discriminator and Generator networks using the standard GAN loss, and training the Qrator network to maximize the mutual information between the generated images and the latent codes. The model is trained for a fixed number of epochs, with the Generator and Qrator networks updated more frequently than the Discriminator network.

## Acknowledgements
This implementation is based on the following paper:
[1] Chen, X., Duan, Y., Houthooft, R., Schulman, J., Sutskever, I., & Abbeel, P. (2016). Infogan: Interpretable representation learning by information maximizing generative adversarial nets. In Advances in Neural Information Processing Systems (pp. 2172-2180).
