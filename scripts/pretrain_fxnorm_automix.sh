#bash
#!/usr/bin/bash

# Script to train automix nets

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

export RESULTS_FOLDER="../trainings" # "/path/to/folder/trainings"
export CONFIGS_FOLDER="../configs/ISMIR" # "/path/to/folder/configs"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# set folder suffix 
export FOLDER_SUFFIX="pretrain" # Example of training naming 

# describe experiment (optional)
export DESCRIPTION="pretrain"

# MODELS_FOLDER="../trainings/results"
# NET='ours_pretrained' # pretrain weights

# train network 
python ../automix/train.py ${CONFIGS_FOLDER}/${FOLDER_SUFFIX}.py \
                --folder-suffix $FOLDER_SUFFIX                 \
                --results-folder $RESULTS_FOLDER               \
                --description $DESCRIPTION &> ${RESULTS_FOLDER}/logs/${FOLDER_SUFFIX}.log < /dev/null &