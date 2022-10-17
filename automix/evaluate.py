"""
Evaluate an automix model.

AI Music Technology Group, Sony Group Corporation
"""

import os

import sys
import numpy as np
import scipy
import argparse
sys.setrecursionlimit(int(1e6))

import torch
import torch.nn as nn

import pyloudnorm as pyln
import sklearn
import librosa

import warnings

from automix.common_dataprocessing import load_wav, save_wav
from automix.common_miscellaneous import uprint
from automix.common_dataprocessing import create_dataset_mixing, create_minibatch_mixing, generate_data
from automix.common_supernet import SuperNet
from automix.common_losses import StereoLoss, StereoLoss2, Loss
from automix import utils

from automix.utils_data_normalization import get_comp_matching, get_eq_matching, get_mean_peak
from automix.utils_data_normalization import get_panning_matching, get_SPS, amp_to_db
from automix.utils_data_normalization import compute_loudness_features, compute_spectral_features, compute_panning_features
from automix.utils_data_normalization import compute_dynamic_features, print_dict

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool, Pool

import functools
import time
import sox



DEBUG = False

SAVE_OUTPUT_WAV = True

# Compute low-level metrics
COMPUTE_LOUDNESS = False
COMPUTE_SPECTRAL = False
COMPUTE_PANNING = False
COMPUTE_DYNAMIC = False

# Compute datapreprocessing, False if stems have already been preprocessed
COMPUTE_NORMALIZATION = False

# Samples to pad at the start of the stems (to start LSTM states)
PAD_SAMPLES = 30*44100

# Max length input samples in seconds
MAX_LENGTH = (7 * 60)

# IDs of song folders to save mixes
SAVE_IDX = [0,1,5,7,8,9,13,14,15,16]


# Costants for effect-normalization preprocessing
CPU_COUNT = cpu_count()//2
FFT_SIZE = 2**16
HOP_LENGTH = FFT_SIZE//4
DTYPE = np.int16
NTAPS = 1001
LUFS = -30
MIN_DB = -50 # Min amplitude to apply EQ matching
MIN_DB_f = -10
NORM_PANNING = True
PANNING_GAIN_K = 0.9
PANNING_FRAMES = True
MAX_FREQ_PAN = 16000

MIN_DB_f = -10

if PANNING_FRAMES:
    FFT_SIZE_PAN = 2**11
    HOP_LENGTH_PAN = FFT_SIZE_PAN//2
else:
    FFT_SIZE_PAN = 2**16
    HOP_LENGTH_PAN = FFT_SIZE_PAN//4
    
STEMS = ['vocals','drums', 'bass', 'other']

MAX_FREQ_PAN = 16000

MIN_DB_f = -10

COMP_PEAK_NORM = -10.0
COMP_TRUE_PEAK = False
COMP_PERCENTILE = 75 # features_mean (v1) was done with 25
COMP_MIN_TH = -40
COMP_MAX_RATIO = 20
COMP_USE_EXPANDER = False
comp_settings = {key:{} for key in STEMS}
for key in comp_settings:
    if key == 'vocals':
        comp_settings[key]['attack'] = 7.5
        comp_settings[key]['release'] = 400.0
        comp_settings[key]['ratio'] = 4
        comp_settings[key]['n_mels'] = 128
    elif key == 'drums':
        comp_settings[key]['attack'] = 10.0
        comp_settings[key]['release'] = 180.0
        comp_settings[key]['ratio'] = 6
        comp_settings[key]['n_mels'] = 128
    elif key == 'bass':
        comp_settings[key]['attack'] = 10.0
        comp_settings[key]['release'] = 500.0
        comp_settings[key]['ratio'] = 5
        comp_settings[key]['n_mels'] = 16
    elif key == 'other':
        comp_settings[key]['attack'] = 15.0
        comp_settings[key]['release'] = 666.0
        comp_settings[key]['ratio'] = 4
        comp_settings[key]['n_mels'] = 128

if PANNING_FRAMES:
    FFT_SIZE_PAN = 2**11
    HOP_LENGTH_PAN = FFT_SIZE_PAN//2
else:
    FFT_SIZE_PAN = 2**16
    HOP_LENGTH_PAN = FFT_SIZE_PAN//4
    
EFFECTS = ['reverb', 'eq', 'compression', 'panning', 'loudness']



def normalize_audio(args_):
    """
    Effect Normalizes audio input

    Args:
        tuple: audio x list of effects to normalize x source (e.g., vocals) x dataset features

    Returns:
        normalized audio
    """

    audio = args_[0]
    effects = args_[1]
    src = args_[2]
    features_mean = args_[3]
    
    print(f'{src} normalizing {effects}...')

    all_zeros = not np.any(audio)

    if all_zeros == False:

        audio_track = np.pad(audio, ((FFT_SIZE, FFT_SIZE), (0, 0)), mode='constant')

        assert len(audio_track.shape) == 2  # Always expects two dimensions

        if audio_track.shape[1] == 1:    # Converts mono to stereo with repeated channels
            audio_track = np.repeat(audio_track, 2, axis=-1)    

        output_audio = audio_track.copy()
        
        max_db = amp_to_db(np.max(np.abs(output_audio)))
        
        if max_db > MIN_DB:
            
            for effect in effects:

                if effect == 'eq':

                    for ch in range(audio_track.shape[1]):
                        audio_eq_matched = get_eq_matching(output_audio[:, ch],
                                                           features_mean[effect][src],
                                                           sr=SR,
                                                           n_fft=FFT_SIZE,
                                                           hop_length=HOP_LENGTH,
                                                           min_db=MIN_DB,
                                                           ntaps=NTAPS,
                                                           lufs=LUFS)


                        np.copyto(output_audio[:,ch], audio_eq_matched)

                elif effect == 'compression':

                    assert(len(features_mean[effect][src])==2)

                    for ch in range(audio_track.shape[1]):
                        audio_comp_matched = get_comp_matching(output_audio[:, ch],
                                                               features_mean[effect][src][0], 
                                                               features_mean[effect][src][1],
                                                               comp_settings[src]['ratio'],
                                                               comp_settings[src]['attack'],
                                                               comp_settings[src]['release'],
                                                               sr=SR,
                                                               min_db=MIN_DB,
                                                               min_th=COMP_MIN_TH, 
                                                               comp_peak_norm=COMP_PEAK_NORM,
                                                               max_ratio=COMP_MAX_RATIO,
                                                               n_mels=comp_settings[src]['n_mels'],
                                                               true_peak=COMP_TRUE_PEAK,
                                                               percentile=COMP_PERCENTILE, 
                                                               expander=COMP_USE_EXPANDER)

                        np.copyto(output_audio[:,ch], audio_comp_matched[:, 0])

                elif effect == 'panning':

                    if PANNING_FRAMES:
                        ref_phi = features_mean[effect][src][:-1]
                        R = int(2*(features_mean[effect][src].shape[0]-1)/FFT_SIZE_PAN)
                        ref_phi = ref_phi.reshape(-1, R).mean(axis = 1)
                        ref_phi = np.append(ref_phi, features_mean[effect][src][-1])
                    else:
                        ref_phi = features_mean[effect][src]

                    if src == 'drums':
                        MAX_FREQ_PAN = 6000
                    else:
                        MAX_FREQ_PAN = 16000

                    audio_pan_matched = get_panning_matching(output_audio,
                                                             PANNING_GAIN_K * ref_phi,
                                                             sr=SR,
                                                             n_fft=FFT_SIZE_PAN,
                                                             hop_length=HOP_LENGTH_PAN,
                                                             min_db_f=MIN_DB_f,
                                                             max_freq_pan=MAX_FREQ_PAN,
                                                             frames=PANNING_FRAMES)

                    np.copyto(output_audio, audio_pan_matched)


                elif effect == 'loudness':
                    output_audio = utils.lufs_normalize(output_audio, SR, features_mean[effect][src]) 

        output_audio = output_audio[FFT_SIZE:FFT_SIZE+audio.shape[0]]

    else:
        
        print(f'{src} is only zeros...')
        output_audio = audio 
        
    return output_audio

def smooth_feature(feature_dict_):
    """
    Applies smoothing filter to dataset features

    Args:
        dict: dict containing features

    Returns:
        dict containing smoothed features
    """

    
    for effect in EFFECTS:
        for key in STEMS:
            if effect == 'eq':
                if key in ['other', 'vocals']:
                    f = 401
                else:
                    f = 151
                feature_dict_[effect][key] = scipy.signal.savgol_filter(feature_dict_[effect][key],
                                                                        f, 1, mode='mirror')
            elif effect == 'panning':
                feature_dict_[effect][key] = scipy.signal.savgol_filter(feature_dict_[effect][key],
                                                                        501, 1, mode='mirror')
    return feature_dict_


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation parser')

    parser.add_argument('--input-folder', help='Audio folder to mix', type=str,
                        required=True)

    parser.add_argument('--training-params', type=str,
                        required=True, help='Path to training parameters and audio settings')

    parser.add_argument('--nets', type=str, help='path to net.dump or net.jit.dump files',
                        default=None)

    parser.add_argument('--weights', type=str,
                        help='Path to weigths for corresponding nets',
                        default=None)

    parser.add_argument('--output-folder', type=str,
                        help='Name of output folder',
                        required=True)
    
    parser.add_argument('--output-prefix', type=str,
                        help='Name of output prefix',
                        required=True)
    
    parser.add_argument('--features', type=str,
                        help='File with effect normalization features',
                        default=None)
    
    parser.add_argument('--baseline-sum', type=bool,
                        help='Boolean flag to output baseline sum of stems',
                        default=False)

    parser.add_argument('--wav-format', type=str,
                        help='Store WAV files with separations in specified format. '
                             + 'If not specified, then we store in the same format as the input WAV file.',
                        default=None, choices=['np.int16', 'np.int32'])

    args = parser.parse_args()
    
    start_time_total = time.time()
    real_times = []
    
    path_dataset = args.input_folder
    
    output_folder = args.output_folder
    output_prefix = args.output_prefix
    
    features_path = args.features
    
    config_file = args.training_params
    weights = args.weights
    
    baseline_sum = args.baseline_sum
    
    if None in [features_path, weights, config_file] or baseline_sum:
        baseline_sum = True
        print('Processing sum of input stems...')
    

    # Loads model info
    uprint(f'Loading configuration from {config_file}')
    exec(open(config_file).read())
    
    config['INPUTS'] = ['vocals', 'bass', 'drums', 'other']
    
    config['SOURCES'] = config['INPUTS'] + config['OUTPUTS']

    n_channels = config['N_CHANNELS']
    accepted_sampling_rates = config['ACCEPTED_SAMPLING_RATES']
    SR = min(accepted_sampling_rates)
    seq_length_td = config['TRAINING_SEQ_LENGTH']
    STEMS = []
    for i in config['INPUTS']:
        STEMS.append(i.split('_')[0])
        
    f_guard = int(np.ceil(((config['GUARD_LEFT'] - (config['FFT_SIZE'] - 1) - 1) / config['HOP_LENGTH']) + 1) + 1)
    
    config['TEST_LOSSES'] = OrderedDict() 
    config['TEST_LOSSES']['td_l1'] = Loss(nn.L1Loss(), DataType.TIME_SAMPLES)
    config['TEST_LOSSES']['td_l2'] = Loss(nn.MSELoss(), DataType.TIME_SAMPLES)
    
    for loss in config['TEST_LOSSES'].values():
        loss.to('cuda')

    if baseline_sum == False:
        
        net = torch.load(args.nets)
        net.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage))

        unfolding_params = None
        if config['BATCHED_TEST']:
            unfolding_params = {'window_size': config['TRAINING_SEQ_LENGTH'],
                                'guard_left': config['GUARD_LEFT'],
                                'guard_right': config['GUARD_RIGHT'],
                                'input_type': net.input_type}

        super_net = SuperNet(net,
                         stft_window=torch.from_numpy(config['STFT_WINDOW'].astype(np.float32)),
                         stft_hop_length=config['HOP_LENGTH'],
                         batched_valid=config['BATCHED_TEST'],
                         unfolding_params=unfolding_params,
                         training_length=config['TRAINING_SEQ_LENGTH'],
                         training_batch_size=config['BATCH_SIZE'] // 1,
                         use_amp=config['USE_AMP'])

        # Transfer model to the GPU
        super_net.to('cuda')
        super_net.eval()

        if config['QUANTIZATION_OP'] is not None:
            super_net.quantize(config['QUANTIZATION_OP'], config['QUANTIZATION_BW'])
    
    # Loads stems
    uprint('Loading stems...')
    
    data_valid = []
    data_valid.extend(create_dataset_mixing(path=path_dataset,
                                             accepted_sampling_rates=config['ACCEPTED_SAMPLING_RATES'],
                                             sources=STEMS+config['OUTPUTS'],
                                             mapped_sources=config['MAPPED_SOURCES'],
                                             n_channels=config['N_CHANNELS'],
                                             load_to_memory=True,
                                             pad_wrap_samples=PAD_SAMPLES,
                                             debug=DEBUG)[0])
    
    
    
    audio_folders_ = utils.getFilesPath(path_dataset, "*.wav")
    song_names = []
    for p in audio_folders_:
        song_names.append(os.path.dirname(p))
    song_names = set(song_names)
    audio_folders_ = []
    for p in song_names:
        audio_folders_.append(p.split('/')[-1])
    audio_folders_.sort() 

    mixture_outputs = {key:[] for key in [DataType.STFT_MAGNITUDE, DataType.TIME_SAMPLES]}
    mixture_targets = {key:[] for key in [DataType.STFT_MAGNITUDE, DataType.TIME_SAMPLES]}
    
    if DEBUG:
        len_data = 1
    else:
        len_data = len(data_valid)
    for i in range(len_data):
#     for i in SAVE_IDX:
        start_time = time.time()
        
        print('\n', audio_folders_[i], i)
        
        data_key = list(data_valid[i])[0]
        audio_tags = list(data_valid[i])[0].split("-")
        samples = data_valid[i][data_key]()[0].shape[0]
        samples = min(samples, int(MAX_LENGTH*SR))
        data = np.zeros((len(config['OUTPUTS']) + len(config['INPUTS']), 1,
                                samples, config['N_CHANNELS']),
                                dtype=np.float32)  # (1 + S)xB=1xTxC

        stems = {}
        target = {}
        
        print(audio_tags)

        for k, tag in enumerate(audio_tags):
            
            if config['INPUTS'] == config['OUTPUTS']:
                    stems[tag] = data_valid[i][data_key]()[k]
                    target[tag] = data_valid[i][data_key]()[k]
            else:
                if tag in STEMS:
                    stems[tag] = data_valid[i][data_key]()[k]
                elif tag in config['OUTPUTS']:
                    target[tag] = data_valid[i][data_key]()[k]

        for k, out in enumerate(config['OUTPUTS']):
            
            if target[out][:samples].shape[0] < samples:
                print('MIX SHORTER THAN STEMS')
                zeros_ = np.zeros_like(data[k][0])
                zeros_[:target[out][:samples].shape[0]] = target[out][:samples]
                data[k] = zeros_[:samples]
            else:
                data[k] = target[out][:samples]

        for k, inp in enumerate(STEMS):
            if stems[inp][:samples].shape[0] < samples:
                print('MIX LARGER THAN STEMS')
                zeros_ = np.zeros_like(data[k+len(config['OUTPUTS'])][0])
                zeros_[:stems[inp][:samples].shape[0]] = stems[inp][:samples]
                data[k+len(config['OUTPUTS'])] = zeros_[:samples]
            else:
                data[k+len(config['OUTPUTS'])] = stems[inp][:samples]
                
    
        new_samples = (1 + samples//config['KERNEL_SIZE_ENCODER'])*config['KERNEL_SIZE_ENCODER']
        data = np.pad(data,[(0,0), (0,0), (0,new_samples-samples), (0,0)])


        if baseline_sum == False:

            if COMPUTE_NORMALIZATION:
                # Applies effect normalization
                features_mean = np.load(features_path, allow_pickle='TRUE')[()]
                features_mean = smooth_feature(features_mean)

                _func_args = []
                for k, inp in enumerate(STEMS):
                    _func_args.append((data[k+1][0], EFFECTS, inp, features_mean))

                pool = Pool(len(STEMS))
                output_audio = pool.map(normalize_audio, _func_args)

                for k, inp in enumerate(STEMS):
                    data[k+1][0] = output_audio[k]

            # Runs model and saves output audio
            test_data = torch.from_numpy(data)

            with torch.no_grad():

                # move the input data to the GPUs
                test_data = test_data.to(f'cuda:{0}') 

#                 test_out = super_net.inference(test_data)
                
                test_tar, test_out, test_losses_ = super_net.evaluate(test_data, config['TEST_LOSSES'])
                

                audio_out = test_out[DataType.TIME_SAMPLES].cpu().numpy()
                audio_out = audio_out[..., :samples]
                if PAD_SAMPLES:
                    audio_out = audio_out[..., PAD_SAMPLES:]
                
                tar_t = test_tar[DataType.TIME_SAMPLES].cpu().numpy()
                tar_t = tar_t.T[..., 0, 0]
                out_t = test_out[DataType.TIME_SAMPLES].cpu().numpy()
                out_t = out_t.T[..., 0, 0]
                
                tfm = sox.Transformer()
                tfm.silence(location=1)
                tar_t = tfm.build_array(input_array=tar_t, sample_rate_in=SR)
                out_t = out_t[np.abs(out_t.shape[0]-tar_t.shape[0]):, :]
                assert tar_t.shape == out_t.shape
                tfm = sox.Transformer()
                tfm.silence(location=-1)
                tar_t = tfm.build_array(input_array=tar_t, sample_rate_in=SR)
                out_t = out_t[:-np.abs(out_t.shape[0]-tar_t.shape[0]), :]
                assert tar_t.shape == out_t.shape

                mixture_targets[DataType.TIME_SAMPLES].append(tar_t)
                mixture_outputs[DataType.TIME_SAMPLES].append(out_t)        

                audio_output_save = config['OUTPUTS']

                for k, out in enumerate(audio_output_save):
                    audio = audio_out[k, 0, ...]
                    if i in SAVE_IDX and SAVE_OUTPUT_WAV:
                        output_name = os.path.join(output_folder, audio_folders_[i])
                        os.makedirs(output_name, exist_ok=True)
                        output_name = os.path.join(output_name,f'{output_prefix}_mixture.wav')
                        save_wav(output_name, SR, audio.T)
                        print(f'Saving {output_name}')
                    del audio

                torch.cuda.empty_cache()
                del test_tar, test_out, test_losses_, test_data

        else:
            if SAVE_OUTPUT_WAV:
                audio_out = np.sum(data[1:], axis=0)[0]
                audio_out = audio_out[:max_samples, :]
                if PAD_SAMPLES:
                    audio_out = audio_out[PAD_SAMPLES:, :]
                output_name = os.path.join(output_folder, audio_folders_[i])
                os.makedirs(output_name, exist_ok=True)
                output_name = os.path.join(output_name,f'{output_prefix}_baseline_mixture.wav')
                save_wav(output_name, SR, audio_out)

        t = (time.time() - start_time)
        t = '{:.2f}'.format(t)
        length = audio_out.shape[-1]/SR
        real_times.append(float(t)/length)
        length = '{:.2f}'.format(length)
        print(f'--- It took {t} seconds ---')
        print(f'--- to mix {length} seconds ---')
        
    torch.cuda.empty_cache()
    del super_net, net, data_valid, data
    del audio_out
    

    if COMPUTE_LOUDNESS:
              
        start_time = time.time()
        
        print('Computing loudness features...')
        
        loudness = {key:[] for key in ['d_lufs', 'd_peak',]}
        
        _func_args = []
        
        for i, audio_ in enumerate(zip(mixture_outputs[DataType.TIME_SAMPLES],
                                      mixture_targets[DataType.TIME_SAMPLES])):
            _func_args.append((audio_[0],
                               audio_[1],
                               i, SR))

        pool = Pool(CPU_COUNT)
        loudness_ = pool.map(compute_loudness_features, _func_args)  
        
        for i in range(len(loudness_)):
            for key in loudness:
                loudness[key].append(loudness_[i][key])
        feature_name = os.path.join(output_folder, f'loudness.npy')
              
        t = (time.time() - start_time)/60
        t = '{:.2f}'.format(t)
        print(f'--- It took {t} mins ---')
        print(f'--- to compute loudness features ---')
        print(f'--- for {len(loudness_)} songs ---')
            
        print(f'Saving {feature_name}')
        np.save(feature_name, loudness)
        
        
    if COMPUTE_SPECTRAL:
              
        start_time = time.time()
        
        print('Computing spectral features...')
        
        spectral = {key:[] for key in ['centroid_mean',
                                    'bandwidth_mean',
                                    'contrast_l_mean',
                                    'contrast_m_mean',
                                    'contrast_h_mean',
                                    'rolloff_mean',
                                    'flatness_mean',
                                    'mape_mean',

                                      ]}

        _func_args = []
        
        for i, audio_ in enumerate(zip(mixture_outputs[DataType.TIME_SAMPLES],
                                      mixture_targets[DataType.TIME_SAMPLES])):

            _func_args.append((audio_[0],
                               audio_[1],
                               i, SR, config['FFT_SIZE'], config['HOP_LENGTH'], config['N_CHANNELS']))

        pool = Pool(CPU_COUNT)
        spectral_ = pool.map(compute_spectral_features, _func_args)  
        
        for i in range(len(spectral_)):
            for key in spectral:
                spectral[key].append(spectral_[i][key])
        feature_name = os.path.join(output_folder, f'spectral.npy')
              
        t = (time.time() - start_time)/60
        t = '{:.2f}'.format(t)
        print(f'--- It took {t} mins ---')
        print(f'--- to compute spectral features ---')
        print(f'--- for {len(spectral_)} songs ---')
        print(f'Saving {feature_name}')
        np.save(feature_name, spectral)
        
    if COMPUTE_PANNING:
              
        start_time = time.time()
        
        print('Computing panning features...')
        
        panning = {key:[] for key in ['P_t_mean',
                               'P_l_mean',
                               'P_m_mean',
                               'P_h_mean',
                                      'mape_mean',
                                     ]}
        
        _func_args = []
        
        n_fft = 2**11
        hop_length = n_fft//2
        
        for i, audio_ in enumerate(zip(mixture_outputs[DataType.TIME_SAMPLES],
                                      mixture_targets[DataType.TIME_SAMPLES])):


            _func_args.append((audio_[0],
                               audio_[1],
                               i, SR, n_fft, hop_length))

        pool = Pool(CPU_COUNT)
        panning_ = pool.map(compute_panning_features, _func_args)  
        
        for i in range(len(panning_)):
            for key in panning:
                panning[key].append(panning_[i][key])
        feature_name = os.path.join(output_folder, f'panning.npy')
              
        t = (time.time() - start_time)/60
        t = '{:.2f}'.format(t)
        print(f'--- It took {t} mins ---')
        print(f'--- to compute panning features ---')
        print(f'--- for {len(panning_)} songs ---')
            
        print(f'Saving {feature_name}')
        np.save(feature_name, panning)
        
        
    if COMPUTE_DYNAMIC:
              
        start_time = time.time()
        
        print('Computing dynamic features...')
        
        dynamic = {key:[] for key in ['rms_mean',
                               'dyn_mean',
                               'crest_mean',
                               'l_ratio_mean_mape',
                               'l_ratio_mean_l2',
                               'mape_mean',
                                     ]}
        
        _func_args = []
        
        for i, audio_ in enumerate(zip(mixture_outputs[DataType.TIME_SAMPLES],
                                      mixture_targets[DataType.TIME_SAMPLES])):


            _func_args.append((audio_[0],
                               audio_[1],
                               i, SR, config['FFT_SIZE'], config['HOP_LENGTH']))

        pool = Pool(CPU_COUNT)
        dynamic_ = pool.map(compute_dynamic_features, _func_args)  
        
        for i in range(len(dynamic_)):
            for key in dynamic:
                dynamic[key].append(dynamic_[i][key])
        feature_name = os.path.join(output_folder, f'dynamic.npy')
              
        t = (time.time() - start_time)/60
        t = '{:.2f}'.format(t)
        print(f'--- It took {t} mins ---')
        print(f'--- to compute dynamic features ---')
        print(f'--- for {len(dynamic_)} songs ---')
            
        print(f'Saving {feature_name}')
        np.save(feature_name, dynamic)
        
    if COMPUTE_LOUDNESS:    
        print('\n Loudness \n')
        print_dict(loudness)
    if COMPUTE_SPECTRAL:
        print('\n Spectral \n')
        print_dict(spectral)
    if COMPUTE_PANNING:
        print('\n Panning \n')
        print_dict(panning)
    if COMPUTE_DYNAMIC:
        print('\n Dynamic \n')
        print_dict(dynamic)
        
              
        
              
    t = (time.time() - start_time_total)/3600
    t = '{:.2f}'.format(t)
    print(f'--- FINISHED - It took {t} hours ---')
    print(f'--- to mix and compute all features ---')
    print(f'--- for {len_data} songs ---')
    del real_times[0]
    real_time = '{:.4f}'.format(np.mean(real_times))
    print(f'--- Mean real-time ratio - {real_time} ---')  

        

        




        




        

        




        



