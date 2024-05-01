# Verb Conjugation in Transformers Is Determined by Linear Encodings of Subject Number

This repository contains code used to run the experiments in the paper [_Verb Conjugation in Transformers Is 
Determined by Linear 
Encodings of 
Subject Number_](https://arxiv.org/abs/2310.15151) by Sophie Hao and Tal Linzen. 

The experiments are run using the script `run_experiment.py`. The experiments can be run with or without a GPU, 
though running without a GPU is not recommended.

Currently, only BERT models are supported.

## Quick Start

To run an experiment

- using `bert-base-uncased`
- with 5 trials
- using local intervention
- with the subject number subspace estimated from the `[MASK]` token:

`python run_experiment.py -m bert-base-uncased -nt 5 -train mask -test local`

## Dependencies

- tqdm
- NumPy
- SciPy
- scikit-learn
- PyTorch
- 🤗 Transformers
- 🤗 Datasets

## Experiment Description

**Important:** _The terminology in this README file may not match the terminology used in the paper._

This repository contains an NLP experiment designed to determine how information about the phi-features of the subject
of a sentence is encoded in the outputs of BERT encoder layers. Specifically, it tests the hypothesis that the number of
the subject (singular vs. plural) is encoded in an 8-dimensional subspace of the representation space, hereinafter
referred to as the _number space_.

### Procedure

The experiment consists of two steps.

- **"Training":** For each layer of BERT (0–12, where layer 0 is the embedding layer and layers 1–12 are the encoder
  layers), an orthonormal basis for the number space is estimated using the INLP method [
  (Ravfogel et al., 2020)](https://aclanthology.org/2020.acl-main.647/). 
- **"Testing":** We test BERT on subject–verb agreement ([Linzen et al., 2016](https://aclanthology.org/Q16-1037/); 
  [Goldberg, 2019](https://arxiv.org/abs/1901.05287)). For each layer of BERT, we apply causal intervention by 
  reversing 
  the sign of the components of hidden vectors computed by that layer [(Ravfogel et al., 2021)](https://aclanthology.org/2021.conll-1.15/), 
  hypothesizing that this would degrade performance on subject–verb agreement.

### Conditions

The experiment is hyperparameterized by the following conditions.
- **Model:** The name of the BERT model to run the experiment on, as listed in the 
[🤗 Transformers library](https://huggingface.co/docs/transformers/index). The model must be compatible with the 
  [`BertForMaskedLM` class](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM). 
  This hyperparameter is represented by the command line argument `-m`.
- **Number of Trials:** The number of trials of the experiment to run, represented by the command line argument `-nt`.
- **Training Condition:** Represented by the command line argument `-train`, this hyperparameter determines how 
  training data for the linear-SVM probe in the Training step will be obtained. Options are:
    - `subj`: Estimate the number space from subject vectors.
    - `mask`: Estimate the number space from `[MASK]` vectors.
    - `is_are`: Estimate the number space from vectors for the main verb (_is_ or _are_).
    - `random`: Use a random basis for the number space.
- **Testing Condition:** Represented by the command line argument `-test`, this hyperparameter determines which 
  positions of the input sequence will undergo causal intervention during the Testing step. Options are:
    - `local`: Use local invervention.
    - `global`: Use global intervention.
    - `verb`: Apply intervention to the main subject and the embedded verb.
    - `mask`: Apply intervention to `[MASK]`.
    - `control`: Apply intervention to number-neutral words and measure its effect on perplexity.