# automix

Automatic Mixing Project Repository

## Installation

`python setup.py install `


## Training models

### ours


  ```bash
    #!/usr/bin/bash
    # Script to train automix nets

    export RESULTS_FOLDER="/path/to/folder/trainings"
    export CONFIGS_FOLDER="/path/to/folder/configs"

    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=1
    
    # set folder suffix (optional)
    export FOLDER_SUFFIX="ours_S_Lb" # Example of training naming 
    
    # describe experiment (optional)
    export DESCRIPTION="ours_S_Lb"
    
    MODELS_FOLDER="/path/to/folder/trainings/results/ISMIR"
    NET='OURS_PRETRAIN' # pretrain weights
    
    # train network 
    python automix/train.py ${CONFIGS_FOLDER}/${FOLDER_SUFFIX}.py \
                    --folder-suffix $FOLDER_SUFFIX                 \
                    --results-folder $RESULTS_FOLDER               \
                    --weight-initialization ${MODELS_FOLDER}/${NET}/current_model_for_mixture.params \
                    --description $DESCRIPTION &> ${RESULTS_FOLDER}/logs/${FOLDER_SUFFIX}.log < /dev/null &
```

### WAVE-U-Net


  ```bash
    #!/usr/bin/bash
    # Script to train automix nets

    export RESULTS_FOLDER="/path/to/folder/trainings"
    export CONFIGS_FOLDER="/path/to/folder/configs"

    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=1

    # set folder suffix (optional)
    export FOLDER_SUFFIX="wun_S_Lb"

    # describe experiment (optional)
    export DESCRIPTION="wun_S_Lb"

    # train network 
    python automix/train.py ${CONFIGS_FOLDER}/${FOLDER_SUFFIX}.py \
                    --folder-suffix $FOLDER_SUFFIX                 \
                    --results-folder $RESULTS_FOLDER               \
                    --description $DESCRIPTION &> ${RESULTS_FOLDER}/logs/${FOLDER_SUFFIX}.log < /dev/null &
```

## Evaluate models


  ```bash
    #!/usr/bin/bash
    # Script to evaluate automix nets

    export CUDA_VISIBLE_DEVICES=0

    OUTPUT_FOLDER="/path/to/folder/mixes" # Folder name where mixes will be created
    INPUT_FOLDER="/path/to/folder/test_mixes" # Folder name where test dataset is located

    mkdir "${OUTPUT_FOLDER}"

    MODELS_FOLDER="/path/to/folder/trainings/results/ISMIR" # Path to models
    PATH_FEATURES="/path/to/folder/trainings/results/features/MUSDB18_features" # Path to average features file

    NET_="ours_S_Lb" # Model name
    ID='ours_S_Lb' # Created mixes audio files will have this ID
    SUB="valid" # Name of subfolder where to store mixes/metrics 

    mkdir "${OUTPUT_FOLDER}"
    mkdir "${OUTPUT_FOLDER}/${NET}"
    mkdir "${OUTPUT_FOLDER}/${NET}/${SUB}"

    # Evaluation for automixing model
    python automix/evaluate.py --input-folder ${INPUT_FOLDER} \
                   --output-folder ${OUTPUT_FOLDER}/${NET}/${SUB} \
                   --output-prefix ${ID} \
                    --training-params ${MODELS_FOLDER}/${NET}/config.py \
                    --nets ${MODELS_FOLDER}/${NET}/net_mixture.dump \
                    --weights ${MODELS_FOLDER}/${NET}/current_model_for_mixture.params \
                    --features ${PATH_FEATURES}/features_mean.npy 
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


  ```bash
  #!/usr/bin/bash
    # Script to train automix nets


    MODELS_FOLDER="/path/to/folder/trainings/results/ISMIR" # Path to models
    PATH_FEATURES="/path/to/folder/trainings/results/features/MUSDB18_features" # Path to average features file
    
    PATH_IR="/data/martinez/audio/automix/ImpulseResponses/Data_ImpulseResponses/IRCAMVerbV3/44100_processed_3000-4000s"
    PATH_PRE_IR="/data/martinez/audio/automix/ImpulseResponses/Data_ImpulseResponses/IRCAMVerbV3/44100_processed_1000-1500s" 
    # Use PATH_PRE_IR only for DRY stems

    export CUDA_VISIBLE_DEVICES=2
    NET="ours_L_Lb"
    
    # Inference for automatic mixing of dry stems
    python automix/inference.py --vocals dry_vocals.wav \
                        --bass dry_bass.wav \
                        --drums dry_drums.wav \
                        --other dry_other.wav \
                        --output mix_from_dry_stems.wav \
                        --training-params ${MODELS_FOLDER}/${NET}/config.py \
                        --impulse-responses ${PATH_IR} \
                        --pre-impulse-responses ${PATH_PRE_IR} \
                        --nets ${MODELS_FOLDER}/${NET}/net_mixture.dump \
                        --weights ${MODELS_FOLDER}/${NET}/best_model_for_mixture_valid_stereo_loss_mean.params \
                        --features ${PATH_FEATURES}/features_mean_v2.npy \

    # Inference for automatic mixing of wet stems
    python automix/inference.py --vocals wet_vocals.wav \
                        --bass wet_bass.wav \
                        --drums wet_drums.wav \
                        --other wet_other.wav \
                        --output mix_from_wet_stems.wav \
                        --training-params ${MODELS_FOLDER}/${NET}/config.py \
                        --impulse-responses ${PATH_IR} \
                        --nets ${MODELS_FOLDER}/${NET}/net_mixture.dump \
                        --weights ${MODELS_FOLDER}/${NET}/best_model_for_mixture_valid_stereo_loss_mean.params \
                        --features ${PATH_FEATURES}/features_mean_v2.npy \

    # Inference to obtain summation of stems
    python automix/inference.py --vocals vocals.wav \
                        --bass bass.wav \
                        --drums drums.wav \
                        --other other.wav \
                        --output sum.wav \
                        --training-params ${MODELS_FOLDER}/${NET}/config.py \
                        --baseline-sum True
                                
                    

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
* torch>=1.9.0

Please see [requirements.txt](http://link/to/requirements.txt)

