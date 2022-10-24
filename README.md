# Automatic music mixing with deep learning and out-of-domain data

Music mixing traditionally involves recording instruments in the form of clean, individual tracks and blending them into a final mixture using audio effects and expert knowledge (e.g., a mixing engineer). The automation of music production tasks has become an emerging field in recent years, where rule-based methods and machine learning approaches have been explored. Nevertheless, the lack of dry or clean instrument recordings limits the performance of such models, which is still far from professional human-made mixes. We explore whether we can use out-of-domain data such as wet or processed multitrack music recordings and repurpose it to train supervised deep learning models that can bridge the current gap in automatic mixing quality. To achieve this we propose a novel data preprocessing method that allows the models to perform automatic music mixing. We also redesigned a listening test method for evaluating music mixing systems. We validate our results through such subjective tests using highly experienced mixing engineers as participants.

## Installation

`python setup.py install`
`pip install -r requirements.txt`

## Training models



### Pre-train FXNorm-automix

```
  bash scripts/pretrain_fxnorm_automix.sh
```

### FXNorm-automix


   ```
  bash scripts/train_fxnorm_automix.sh
```

   ```
  bash scripts/train_fxnorm_automix.sh
```

### WAVE-U-Net


   ```
  bash scripts/train_wun.sh
```



## Evaluate models


   ```
  bash scripts/evaluate.sh
```

## Models

TDCNFx -> Ours:

  ```
  automix/common_networkbuilding_cafx_tdcn_lstm_mix.py
```

WAVE-U-NET:

  ```
  automix/common_networkbuilding_waveunet.py
```

## Data preprocessing

  ```
  automix/data_normalization.py
```

## Inference 


    ```
  bash scripts/inference.sh
```
                                
                    

```
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

Please see [requirements.txt](http://link/to/requirements.txt)

