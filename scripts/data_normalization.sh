#bash        
#!/usr/bin/bash
# Script to evaluate automix nets

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

INPUT_FOLDER='/path/to/MUSDB18/train' # Folder that contains training dataset to be normalized and where features are computed
INPUT_FOLDER_VAL='/path/to/MUSDB18/val' # Folder that contains validation dataset

# The following are the paths to the folders containing the impulse responses. 
# The data loader expects each IR to be in an individual folder and named impulse_response.wav
# e.g. /path/to/IR/impulse-response-001/impulse_response.wav
# Stereo or Mono are supported
PATH_IR="/path/to/IR" # IRs for data augmentation
PATH_PRE_IR="/path/to/PRE_IR" # IRs for data pre-augmentation if the dataset corresponds to dry stems

PATH_FEATURES="../trainings/features" # Path to average features file

SUFFIX="normalized" # Name suffix to add to the output file name

# Data Normalization. First, average features are computed only (This first pass is only required when we want to apply reverb augmentation)
python ../automix/data_normalization.py --input-folder ${INPUT_FOLDER} \
                    --output-suffix ${SUFFIX} \
                    --features-save ${PATH_FEATURES}/features_MUSDB18_TEST.npy \
                    --normalize False
                    
# Data Normalization. Then, previousy computed features are loaded and normalization is done for 'reverb', 'eq', 'compression', 'panning' and 'loudness' 

# Training set
python ../automix/data_normalization.py --input-folder ${INPUT_FOLDER} \
                    --impulse-responses ${PATH_IR} \
                    --output-suffix ${SUFFIX} \
                    --features-load ${PATH_FEATURES}/features_MUSDB18_TEST.npy \
                    --normalize True
                    
# Validation set
python ../automix/data_normalization.py --input-folder ${INPUT_FOLDER_VAL} \
                    --impulse-responses ${PATH_IR} \
                    --output-suffix ${SUFFIX} \
                    --features-load ${PATH_FEATURES}/features_MUSDB18_TEST.npy \
                    --normalize True                    


# Data Normalization alternative. For example, when reverb augmentation is not needed. Then feature computation and stems normalization can be done in one pass 
# python ../automix/data_normalization.py --input-folder ${INPUT_FOLDER} \
#                     --output-suffix ${SUFFIX} \
#                     --features-save ${PATH_FEATURES}/features_MUSDB18_TEST.npy \
#                     --normalize True
