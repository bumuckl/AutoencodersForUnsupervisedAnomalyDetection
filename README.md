# AutoencodersForUnsupervisedAnomalyDetection
The code behind my paper "Deep Autoeoncoding Models for Unsupervised Anomaly Detection in Brain MR Images"


This is the source code for my paper **Deep Autoeoncoding Models for Unsupervised Anomaly Detection in Brain MR Images** (Baur et al. <https://arxiv.org/abs/1804.04488>), accepted at the MICCAI 2018 BrainLesion Workshop, presented as an oral and as a poster.

The source code comprises my object-oriented Deep-Learning framework, developed on top of TensorFlow, as well as multiple files for training & evaluating the various Auto-Encoders and Generative Adversarial Networks described in the paper:

- DLMODEL.py: the base class for all your TensorFlow Deep Learning needs
- AE.py: inherits from DLMODEL, implements training & inference of Autoencoders
- AEGAN.py: inherits from DLMODEL, implements training & inference of Autoencoders + Adversarial Network
- VAE.py: inherits from DLMODEL, implements training & inference of Variational Autoencoders
- VAEGAN.py: inherits from DLMODEL, implements training & inference of a VAE-GAN
- Losses.py, custom_layers.py, utils.py: Utilities...
- architectures/*: This folder contains network architectures, which can be plugged in any of the above objects for training
- main_*.py: The files that will launch a training and evaluation of a specific model
- Experiment.py: THe file that describes the evaluation pipeline.

The dataset used for this work was a non-public, clinical brain MR dataset with both healthy subjects, and subjects with MS, and is therefore not provided. You are free to plug in your own dataset though, by simply taking a look at the class "Dataset.py". Of great importance is the implementation of the "next_batch(...)" method.

# Training & Evaluation

1. pip install -r requirements.txt
2. python main_*.py
