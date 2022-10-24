#bash        
#!/usr/bin/bash
# Script to evaluate automix nets

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

export CUDA_VISIBLE_DEVICES=1

OUTPUT_FOLDER="../mixes" # Folder name where mixes will be created
# INPUT_FOLDER="/path/to/folder/test_mixes" # Folder name where test dataset is located
INPUT_FOLDER="/data/automix/MUSDB18/test"

CONFIGS_FOLDER="../configs/ISMIR" # "/path to folder with configs files"

# The following are the paths to the folders containing the impulse responses. 
# The data loader expects each IR to be in an individual folder and named impulse_response.wav
# e.g. /path/to/IR/impulse-response-001/impulse_response.wav
# Stereo or Mono are supported
PATH_IR="/path/to/IR" # IRs for data augmentation
PATH_PRE_IR="/path/to/PRE_IR" # IRs for data pre-augmentation of dry stems

mkdir "${OUTPUT_FOLDER}"

MODELS_FOLDER="../trainings/results"
PATH_FEATURES="../trainings/features" # Path to average features file

NET="ours_S_Lb" # Model name
ID='ours_S_Lb' # Created mixes audio files will have this ID
SUB="test_wet" # Name of subfolder where to store mixes/metrics 

mkdir "${OUTPUT_FOLDER}"
mkdir "${OUTPUT_FOLDER}/${NET}"
mkdir "${OUTPUT_FOLDER}/${NET}/${SUB}"

# Evaluation for automixing model (wet stems)
python ../automix/evaluate.py --input-folder ${INPUT_FOLDER} \
            --output-folder ${OUTPUT_FOLDER}/${NET}/${SUB} \
            --output-prefix ${ID} \
            --training-params ${CONFIGS_FOLDER}/${NET}.py \
            --impulse-responses ${PATH_IR} \
            --nets ${MODELS_FOLDER}/${NET}/net_mixture.dump \
            --weights ${MODELS_FOLDER}/${NET}/best_model_for_mixture_valid_stereo_loss_mean.params \
            --features ${PATH_FEATURES}/features_MUSDB18.npy
            
# # Evaluation for automixing model (dry stems)
# # python ../automix/evaluate.py --input-folder ${INPUT_FOLDER} \
#             --output-folder ${OUTPUT_FOLDER}/${NET}/${SUB} \
#             --output-prefix ${ID} \
#             --training-params ${CONFIGS_FOLDER}/${NET}.py \
#             --impulse-responses ${PATH_IR} \
#             --pre-impulse-responses ${PATH_PRE_IR} \
#             --nets ${MODELS_FOLDER}/${NET}/net_mixture.dump \
#             --weights ${MODELS_FOLDER}/${NET}/best_model_for_mixture_valid_stereo_loss_mean.params \
#             --features ${PATH_FEATURES}/features_MUSDB18.npy
           
# # Evaluation for summation of Stems (Normalized Baseline)
# python ../automix/evaluate.py --input-folder ${INPUT_FOLDER} \
#             --output-folder ${OUTPUT_FOLDER}/${NET}/${SUB} \
#             --output-prefix ${ID} \
#             --impulse-responses ${PATH_IR} \
#             --pre-impulse-responses ${PATH_PRE_IR} \
#             --baseline-sum True \
#             --features ${PATH_FEATURES}/features_MUSDB18.npy