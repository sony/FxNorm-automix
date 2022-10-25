# Automatic music mixing with deep learning and out-of-domain data

Music mixing traditionally involves recording instruments in the form of clean, individual tracks and blending them into a final mixture using audio effects and expert knowledge (e.g., a mixing engineer). The automation of music production tasks has become an emerging field in recent years, where rule-based methods and machine learning approaches have been explored. Nevertheless, the lack of dry or clean instrument recordings limits the performance of such models, which is still far from professional human-made mixes. We explore whether we can use out-of-domain data such as wet or processed multitrack music recordings and repurpose it to train supervised deep learning models that can bridge the current gap in automatic mixing quality. To achieve this we propose a novel data preprocessing method that allows the models to perform automatic music mixing. We also redesigned a listening test method for evaluating music mixing systems. We validate our results through such subjective tests using highly experienced mixing engineers as participants.

## Paper

For technical details of the work, please see:

"[Automatic music mixing with deep learning and out-of-domain data.](https://arxiv.org/abs/2208.11428)"
[Marco A. Martínez-Ramírez](https://m-marco.com/about/), [Wei-Hsiang Liao](https://jp.linkedin.com/in/wei-hsiang-liao-66283154), [Giorgio Fabbro](https://twitter.com/GioFabbro), [Stefan Uhlich](https://scholar.google.de/citations?user=hja8ejYAAAAJ&hl=de), [Chihiro Nagashima](https://jp.linkedin.com/in/chihiro-nagashima-9473271aa), and [Yuki Mitsufuji](https://www.yukimitsufuji.com/). <br />
23rd International Society for Music Information Retrieval Conference (ISMIR), December, 2022.

```
@inproceedings{martinez2022FxNormAutomix,
   title={Automatic music mixing with deep learning and out-of-domain data},
   author={Mart\'{i}nez-Ram\'{i}rez, Marco A. and Liao, Wei-Hsiang and Fabbro, Giorgio and Uhlich, Stefan and Nagashima, Chihiro and Mitsufuji, Yuki},
   booktitle={23rd International Society for Music Information Retrieval Conference (ISMIR)},<br />
   month={December},
   year={2022}
}
```
Main Project Page: https://marco-martinez-sony.github.io/FxNorm-automix/

ArXiv Paper: https://arxiv.org/abs/2208.11428

## Installation

```
  python setup.py install
```
```
  pip install -r requirements.txt
```

## Data preprocessing - Effect Normalization

This script computes the average features for the training dataset, then normalizes the train and validation partitions.

```
  bash scripts/data_normalization.sh
```   

## Training models

The following scripts train FxNorm-Automix (Ours) and Wave-U-Net (WUN) models. Check `config.py` files at `configs/ISMIR` for hyperparameters and dataset settings

### Pre-train FXNorm-automix

```
  bash scripts/pretrain_fxnorm_automix.sh
```

### FXNorm-automix


```
  bash scripts/train_fxnorm_automix.sh
```


### WAVE-U-Net


```
  bash scripts/train_wun.sh
```



## Evaluate models

This script evaluates a trained model on a given test dataset; mixes and metrics are computed

```
  bash scripts/evaluate.sh
```


## Inference 

This script runs inference on a given multitrack 

```
  bash scripts/inference.sh
```
                          
## Trained Models

Trained models can be found at `training/results` and their `config.py` files at `configs/ISMIR`

The available models are *ours_S_La*, *ours_S_Lb*, *ours_S(pretrained)*, and *wun_S_Lb*

## Computed features

The average features computed on MUSDB18 can be found at `training/features/features_MUSDB18.npy`

                 

## Impulse Responses (IR)

Due to copyright issues, the IRs used during training, evaluation and inference of our models cannot be made public.

However, users can provide their IRs for data augmentation. Ideally, the reverberation time (RT) of the IRs should be between 2 and 4 seconds. For "pre-reverb" IRs (when doing inference on real dry data), RT should be less than 1.5 seconds. (Stereo and Mono are supported)

The data loader expects each IR to be in an individual folder and named impulse_response.wav
e.g. /path/to/IR/impulse-response-001/impulse_response.wav 

## Requirements

* librosa>=0.8.1
* psutil
* `pymixconsole` (`pip install git+https://github.com/csteinmetz1/pymixconsole`)
* `pyloudnorm` (`pip install git+https://github.com/csteinmetz1/pyloudnorm`)
* `aubio` (`pip install git+https://github.com/aubio/aubio`)
* scipy>=1.6.3
* soundfile
* soxbindings
* sty
* tensorboard
* setuptools==59.5.0
* torch==1.9.0

Please see [requirements.txt](https://github.com/sony/FxNorm-automix/blob/main/requirements.txt)

