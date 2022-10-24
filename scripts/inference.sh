#bash        
#!/usr/bin/bash
# Script to evaluate automix nets

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

export CUDA_VISIBLE_DEVICES=1

OUTPUT_FOLDER="../mixes" # Folder name where mixes will be created
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

# Inference for automatic mixing of dry stems
python ../automix/inference.py --vocals dry_vocals.wav \
                    --bass dry_bass.wav \
                    --drums dry_drums.wav \
                    --other dry_other.wav \
                    --output ../mixes/mix_from_dry_stems.wav \
                    --training-params ${CONFIGS_FOLDER}/${NET}.py \
                    --impulse-responses ${PATH_IR} \
                    --pre-impulse-responses ${PATH_PRE_IR} \
                    --nets ${MODELS_FOLDER}/${NET}/net_mixture.dump \
                    --weights ${MODELS_FOLDER}/${NET}/best_model_for_mixture_valid_stereo_loss_mean.params \
                    --features ${PATH_FEATURES}/features_MUSDB18.npy 

# # Inference for automatic mixing of wet stems
# python ../automix/inference.py --vocals wet_vocals.wav \
#                     --bass wet_bass.wav \
#                     --drums wet_drums.wav \
#                     --other wet_other.wav \
#                     --output ../mixes/mix_from_wet_stems.wav \
#                     --training-params ${CONFIGS_FOLDER}/${NET}.py \
#                     --impulse-responses ${PATH_IR} \
#                     --nets ${MODELS_FOLDER}/${NET}/net_mixture.dump \
#                     --weights ${MODELS_FOLDER}/${NET}/best_model_for_mixture_valid_stereo_loss_mean.params \
#                     --features ${PATH_FEATURES}/features_MUSDB18.npy \

# # Inference to obtain summation of normalized stems
# python ../automix/inference.py --vocals vocals.wav \
#                     --bass bass.wav \
#                     --drums drums.wav \
#                     --other other.wav \
#                     --output ../mixes/sum.wav \
#                     --impulse-responses ${PATH_IR} \
#                     --pre-impulse-responses ${PATH_PRE_IR} \
#                     --features ${PATH_FEATURES}/features_MUSDB18.npy \
#                     --baseline-sum True

