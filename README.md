# Rethinking Pruning for Accelerating Deep Inference At the Edge

This repository contains code of the paper, [Rethinking Pruning for Accelerating Deep Inference At the Edge (Gao et al., KDD 2020)](https://dl.acm.org/doi/abs/10.1145/3394486.3403058). It includes code for training and pruning models with VIBNet and MNP methods in Automatic Speech Recognition(ASR) and Neural Entity Recognition(NER) applications.



### Dependencies

This code requires

 - python 3.*
 - TensorFlow v1.13



### Data

For the dataset, you can refer to [CoNLL-2003](https://www.aclweb.org/anthology/W03-0419/) and [Librispeech](https://ieeexplore.ieee.org/document/7178964/). 

For data pre-processing (MFCC features) and decoding in Librispeech, pleasse see [pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi). In this repository, the MFCC features are supposed to be stored in csv files, also you can rewrite `data_loader/data_loader.py` to load your data.



### Usage

To run the code for NER and ASR, see the usage instructions in `models/bilstm_ner.py`, `models/mnp_ner.py`, `models/lstm_asr.py` and  `models/mnp_asr.py`.

