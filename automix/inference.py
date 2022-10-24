"""
Apply a trained network to mix Vocals, Bass, Drums and Other stems.

TL21
"""

import os

import sys
import numpy as np
import scipy
import argparse
sys.setrecursionlimit(int(1e6))

import torch
import torch.nn as nn

import soundfile as sf
import librosa

from automix.common_dataprocessing import create_dataset, load_wav, save_wav
from automix.common_miscellaneous import uprint
from automix.common_dataprocessing import create_dataset_mixing, create_minibatch_mixing, generate_data
from automix.common_supernet import SuperNet
from automix import utils

from automix.utils_data_normalization import get_comp_matching, get_eq_matching, get_mean_peak, lufs_normalize
from automix.utils_data_normalization import get_panning_matching, get_SPS, amp_to_db, get_reverb_send

from multiprocessing.pool import ThreadPool, Pool

from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList

import functools
import time



# Cosntants for effect-normalization preprocessing

EFFECTS = ['prereverb', 'reverb', 'eq', 'compression', 'panning', 'loudness']

# Compute datapreprocessing, False if stems have already been preprocessed
COMPUTE_NORMALIZATION = True

# General Settings
CPU_COUNT = 4
SUBTYPE = 'PCM_16'
FFT_SIZE = 2**16
HOP_LENGTH = FFT_SIZE//4

# Specific Audio Effect Settings

# EQ
NORM_EQ = True
NTAPS = 1001
LUFS = -30
MIN_DB = -50 # Min amplitude to apply EQ matching

# Panning
NORM_PANNING = True
PANNING_GAIN_K = 0.9 # Empirical results show panning offset shows better re-panning
PANNING_FRAMES = True
MAX_FREQ_PAN = 16000
MIN_DB_f = -10
if PANNING_FRAMES:
    FFT_SIZE_PAN = 2**11
    HOP_LENGTH_PAN = FFT_SIZE_PAN//2
else:
    FFT_SIZE_PAN = 2**16
    HOP_LENGTH_PAN = FFT_SIZE_PAN//4

# Compressor
NORM_COMP = True
COMP_USE_EXPANDER = False
COMP_PEAK_NORM = -10.0
COMP_TRUE_PEAK = False
COMP_PERCENTILE = 75 
COMP_MIN_TH = -40
COMP_MAX_RATIO = 20
comp_settings = {key:{} for key in ['vocals', 'drums', 'bass', 'other']}
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

# REVERB

EQ_PARAMETERS = ParameterList()
eq_gain = -30.0
EQ_PARAMETERS.add(Parameter('low_shelf_gain', eq_gain, 'float', minimum=eq_gain, maximum=eq_gain))
EQ_PARAMETERS.add(Parameter('low_shelf_freq', 600.0, 'float', minimum=500.0, maximum=700.0))
EQ_PARAMETERS.add(Parameter('high_shelf_gain', eq_gain, 'float', minimum=eq_gain, maximum=eq_gain))
EQ_PARAMETERS.add(Parameter('high_shelf_freq', 8500.0, 'float', minimum=7000.0, maximum=10000.0))
EQ_BANDS_VOCALS_OTHER = ['low_shelf', 'high_shelf']

# PRE-REVERB (FOR REAL DRY INPUT)
PRE_REVERB_PARAMETERS = ParameterList()
PRE_REVERB_PARAMETERS.add(Parameter('wet', 0.5, 'float', minimum=0.5, maximum=0.5))
PRE_REVERB_PARAMETERS.add(Parameter('dry', 0.5, 'float', minimum=0.5, maximum=0.5))
PRE_REVERB_PARAMETERS.add(Parameter('decay', 1.0, 'float', minimum=1.0, maximum=1.0))
PRE_REVERB_PARAMETERS.add(Parameter('pre_delay', 0, 'int', units='ms', minimum=0, maximum=0))
# ABBEY ROAD TRICK SETTINGS
PRE_EQ_PROBABILITY = 0
PRE_REVERB_PROBABILITY = 1.0
PRE_FX_PARALLEL = False
PRE_FX_SHUFFLE = False

# REVERB AUGMENTATION 
REVERB_PARAMETERS = ParameterList()
REVERB_PARAMETERS.add(Parameter('wet', 1.0, 'float', minimum=1.0, maximum=1.0))
REVERB_PARAMETERS.add(Parameter('dry', 0.0, 'float', minimum=0.0, maximum=0.0))
REVERB_PARAMETERS.add(Parameter('decay', 1.0, 'float', minimum=1.0, maximum=1.0))
REVERB_PARAMETERS.add(Parameter('pre_delay', 0, 'int', units='ms', minimum=0, maximum=0))
# ABBEY ROAD TRICK SETTINGS
EQ_PROBABILITY = 1.0
REVERB_PROBABILITY = 1.0
FX_PARALLEL = True 
FX_SHUFFLE = False
    

def normalize_audio_wave(args_):
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
                    output_audio = lufs_normalize(output_audio, SR, features_mean[effect][src]) 
                    
                elif effect == 'reverb':
                    
                    if src in ['bass', 'drums']: 
                        pass
                    else:
                        bands = EQ_BANDS_VOCALS_OTHER

                        audio_reverb_send = get_reverb_send(output_audio, EQ_PARAMETERS, REVERB_PARAMETERS, IR,
                                                            bands=bands,
                                                            eq_prob=EQ_PROBABILITY, rv_prob=REVERB_PROBABILITY,
                                                            parallel=FX_PARALLEL,
                                                            shuffle=FX_SHUFFLE,
                                                            sr=SR)


                        np.copyto(output_audio, audio_reverb_send)
                    
                elif effect == 'prereverb':
                    
                    if src in ['bass', 'drums']: 
                        pass
                    else:
                        bands = EQ_BANDS_VOCALS_OTHER
                    
                        audio_reverb_send = get_reverb_send(output_audio, EQ_PARAMETERS, PRE_REVERB_PARAMETERS, PRE_IR,
                                                            bands=bands,
                                                            eq_prob=PRE_EQ_PROBABILITY, rv_prob=PRE_REVERB_PROBABILITY,
                                                            parallel=PRE_FX_PARALLEL,
                                                            shuffle=PRE_FX_SHUFFLE,
                                                            sr=SR)


                        np.copyto(output_audio, audio_reverb_send)

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

    parser.add_argument('--vocals', type=str,
                        default=None, help='Path to vocals')
    
    parser.add_argument('--bass', type=str,
                        default=None, help='Path to bass')
    
    parser.add_argument('--drums', type=str,
                        default=None, help='Path to drums')
    
    parser.add_argument('--other', type=str,
                        default=None, help='Path to other')

    parser.add_argument('--training-params', type=str,
                        default=None, help='Path to training parameters and audio settings')
    
    parser.add_argument('--impulse-responses', type=str,
                        default=None, help='Path to folder with impulse responses')
    
    parser.add_argument('--pre-impulse-responses', type=str, help='Path to folder with impulse responses for pre-reverb',
                        default=None)

    parser.add_argument('--nets', type=str, help='path to net.dump or net.jit.dump files',
                        default=None)

    parser.add_argument('--weights', type=str,
                        help='Path to weigths for corresponding nets',
                        default=None)

    parser.add_argument('--output', type=str,
                        help='Name of output file',
                        required=True)
    
    parser.add_argument('--features', type=str,
                        help='File with effect normalization features',
                        required=True)
    
    parser.add_argument('--baseline-sum', type=bool,
                        help='Boolean flag to output baseline sum of stems',
                        default=False)
    
    start_time = time.time()


    args = parser.parse_args()
    
    audio_path = {} 
    audio_path['vocals'] = args.vocals
    audio_path['bass'] = args.bass
    audio_path['drums'] = args.drums
    audio_path['other'] = args.other
    
    ir_path = {} 
    ir_path['reverb'] = args.impulse_responses
    ir_path['prereverb'] = args.pre_impulse_responses
        
    if not any([args.vocals, args.bass, args.drums, args.other]):
        raise ValueError(f'All inputs are None. Please input at least one stem')
    
    output_name = args.output
    features_path = args.features
    
    config_file = args.training_params
    weights = args.weights
    
    baseline_sum = args.baseline_sum
    
    if None in [features_path, weights, config_file] or baseline_sum:
        baseline_sum = True
        print('Processing sum of input stems...')

    if baseline_sum == False:
        
        # Loads model info
        uprint(f'Loading configuration from {config_file}')
        exec(open(config_file).read())

        n_channels = config['N_CHANNELS']
        accepted_sampling_rates = config['ACCEPTED_SAMPLING_RATES']
        SR = min(accepted_sampling_rates)
        # Samples to pad at the start of the stems (to start LSTM states)
        pad_samples = 30*SR # None if we don't want to pad samples 
        STEMS = []
        for i in config['INPUTS']:
            STEMS.append(i.split('_')[0])
        
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
            
    else:
        
        n_channels = 2
        accepted_sampling_rates = [44100, 48000]
        SR = min(accepted_sampling_rates)
        # Samples to pad at the start of the stems (to start LSTM states)
        pad_samples = None # None if we don't want to pad samples 
        STEMS = ['vocals', 'bass', 'drums', 'other']
    
    # Loads stems
    uprint('Loading stems...')
    
    samples = []
    subtype = []
    samplingrate = []
    audio = [None] * len(STEMS)
    for k, inp in enumerate(STEMS):
        
        if audio_path[inp]:
    
            _audio = load_wav(audio_path[inp], mmap=False)
        
            if pad_samples:
                    _audio = (_audio[0], np.pad(_audio[1], [(pad_samples, 0), (0,0)], 'wrap'))

            # determine properties from loaded data
            _samplingrate = _audio[0]
            _n_channels = _audio[1].shape[1]

            samplingrate.append(_samplingrate)
            ob = sf.SoundFile(audio_path[inp])
            subtype.append(ob.subtype)


            # make sure that sample rate and number of channels matches
            if _n_channels != n_channels:
                if _n_channels == 1:    # Converts mono to stereo with repeated channels
                    _audio = (_audio[0], np.repeat(_audio[1], 2, axis=-1)) 
                    print("Converted file to stereo by repeating mono channel")
                else:
                    raise ValueError(f'File has {_n_channels} '
                                     f'channels but expected {n_channels}.')

            if _samplingrate not in accepted_sampling_rates:
                print(f'File has fs = {_samplingrate}Hz '
                                 f'but expected {accepted_sampling_rates}Hz.')
                print('File will be resampled for processing...')

                temp_audio = _audio[1].copy()
                dtype = _audio[1].dtype

                conversion_scale = 1. / (1. + np.iinfo(temp_audio.dtype).max)
                temp_audio = temp_audio.astype(dtype=np.float32) * conversion_scale

                temp_audio = librosa.resample(temp_audio.T, _samplingrate, SR, res_type='kaiser_best').T

                temp_audio = temp_audio * (1 + np.iinfo(dtype).max)
                if np.min(temp_audio) < np.iinfo(dtype).min or np.max(temp_audio) > np.iinfo(dtype).max:
                    uprint(f'WARNING: Clipping occurs for {file_path}. when resampling')
                temp_audio = np.clip(temp_audio, np.iinfo(dtype).min, np.iinfo(dtype).max)
                temp_audio = temp_audio.astype(dtype)

                _audio = (SR, temp_audio)

            _samples = _audio[1].shape[0]

            samples.append(_samples)
            audio[k] = _audio
            
        else: 
            print(f'File for {inp} is missing, thus mix will be created without this stem.')
            # Creates dummy array for missing stems
            audio[k] = (SR, np.zeros((SR, n_channels), dtype=np.int16))
            
        
    data_func = functools.partial(generate_data, file_path_or_data=audio)    
        
    max_samples = max(samples)
    max_samplingrate = max(samplingrate)
    max_subtype = max(subtype)
    data = np.zeros((len(STEMS), 1,
                            max_samples, n_channels),
                            dtype=np.float32)  # (1 + S)xB=1xTxC
    stems = {}
    for k, inp in enumerate(STEMS):
        stems[inp] = data_func()[k]

    for k, inp in enumerate(STEMS):
        if stems[inp][:max_samples].shape[0] < max_samples:
            print(f'{inp} stem do not have same size as the rest, zero padding...')
            zeros_ = np.zeros_like(data[k][0])
            zeros_[:stems[inp][:max_samples].shape[0]] = stems[inp][:max_samples]
            data[k] = zeros_[:max_samples]
        else:
            data[k] = stems[inp][:max_samples]
    
    if baseline_sum == False:
        new_samples = (1 + max_samples//config['KERNEL_SIZE_ENCODER'])*config['KERNEL_SIZE_ENCODER']
        data = np.pad(data,[(0,0), (0,0), (0,new_samples-max_samples), (0,0)])

    
        
    # Loading Impuse Responses

    # PRE-REVERB
    if ir_path['prereverb']:
        PRE_IR = []
        PRE_IR.extend(create_dataset(path=ir_path['prereverb'],
                                                accepted_sampling_rates=[SR],
                                                sources=['impulse_response'],
                                                mapped_sources={}, load_to_memory=True, debug=False)[0])
        PRE_REVERB_PARAMETERS.add(Parameter('index', 0, 'int', minimum=0, maximum=len(PRE_IR)))
    else:
        EFFECTS.remove('prereverb')

    # Conv Reverb
    if ir_path['reverb']:
        IR = []
        IR.extend(create_dataset(path=ir_path['reverb'],
                                                accepted_sampling_rates=[SR],
                                                sources=['impulse_response'],
                                                mapped_sources={}, load_to_memory=True, debug=False)[0])
        REVERB_PARAMETERS.add(Parameter('index', 0, 'int', minimum=0, maximum=len(IR)))
    else:
        EFFECTS.remove('reverb')
        
    if COMPUTE_NORMALIZATION:
        # Applies effect normalization
        features_mean = np.load(features_path, allow_pickle='TRUE')[()]
        features_mean = smooth_feature(features_mean)

        _func_args = []
        for k, inp in enumerate(STEMS):
            _func_args.append((data[k][0], EFFECTS, inp, features_mean))

        pool = Pool(CPU_COUNT)
        output_audio = pool.map(normalize_audio_wave, _func_args)

        for k, inp in enumerate(STEMS):
            data[k][0] = output_audio[k]
            
    if baseline_sum == False:

        # Runs model and saves output audio
        test_data = torch.from_numpy(data)

        with torch.no_grad():

            # move the input data to the GPUs
            test_data = test_data.to(f'cuda:{0}') 

            test_out = super_net.inference(test_data)

            audio_out = test_out[DataType.TIME_SAMPLES].cpu().numpy()
            audio_out = audio_out[..., :max_samples]
            if pad_samples:
                audio_out = audio_out[..., pad_samples:]

            audio_output_save = config['OUTPUTS']

            for k, out in enumerate(audio_output_save):
                audio = audio_out[k, 0, ...]
                if max_samplingrate not in accepted_sampling_rates:
                    audio = librosa.resample(audio, SR, max_samplingrate, res_type='kaiser_best')
                save_wav(output_name, max_samplingrate, audio.T, subtype=max_subtype)
                del audio

            torch.cuda.empty_cache()
            del test_out, audio_out
            
    else:
        
        audio_out = np.sum(data, axis=0)[0]
        audio_out = audio_out[:max_samples, :]
        if pad_samples:
            audio_out = audio_out[pad_samples:, :]
        if max_samplingrate not in accepted_sampling_rates:
            audio_out = librosa.resample(audio_out.T, SR, max_samplingrate, res_type='kaiser_best').T
        save_wav(output_name, max_samplingrate, audio_out, subtype=max_subtype)

t = (time.time() - start_time)
t = '{:.2f}'.format(t)
length = max_samples/max_samplingrate
length = '{:.2f}'.format(length)
print(f'--- It took {t} seconds ---')
print(f'--- to mix {length} seconds ---')



