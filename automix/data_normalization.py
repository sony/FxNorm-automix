import os
os.environ["OMP_NUM_THREADS"] = "1" 
import sys
import time
import numpy as np
import scipy
import pathlib
import librosa
import pyloudnorm as pyln

sys.setrecursionlimit(int(1e6))

import sklearn

from automix.common_dataprocessing import load_wav, save_wav, create_dataset
from automix.common_miscellaneous import compute_stft, compute_istft
from automix import utils

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool, Pool
from collections import OrderedDict

import soundfile as sf

import time

from automix.utils_data_normalization import get_comp_matching, get_eq_matching, get_mean_peak, lufs_normalize
from automix.utils_data_normalization import get_reverb_send, get_panning_matching, get_SPS, amp_to_db

from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList


PATH_DATASET = '/data/martinez/audio/automix/TencyDB/train' # Folder that contains dataset to be normalized
PATH_FEATURES = '/data/martinez/audio/automix/TencyDB_features++' # Folder that contains average features files
FILE_NAME = 'rv1v0off_eq_comp_noexp_pan_frames_vol' # Prefix to name normalized audio
NAME_FEATURES = 'features_mean_v3' # Name of feature file to be loaded or to be created
STEMS = ['vocals','drums', 'bass', 'other'] # Stems to be normalized
EFFECTS = ['reverb', 'eq', 'compression', 'panning', 'loudness'] # Effects to be normalized, order matters
CPU_COUNT = cpu_count()//2

LOAD_MEAN = True # IF average features are loaded. If false, features are calculated and file is saved
NORM_AUDIO = True # If audio should be normalized. If false only features calculated.

# Audio settings
SR = 44100
SUBTYPE = 'PCM_16'

# General Settings
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
EQ_PARAMETERS.add(Parameter('first_band_gain', eq_gain, 'float', minimum=eq_gain, maximum=eq_gain)) #first_band only applies to drums.
EQ_PARAMETERS.add(Parameter('first_band_freq', 100.0, 'float', minimum=100.0, maximum=100.0))
EQ_PARAMETERS.add(Parameter('first_band_q', 0.7, 'float', minimum=0.7, maximum=0.7))

EQ_BANDS_VOCALS_OTHER = ['low_shelf', 'high_shelf']
EQ_BANDS_DRUMS = ['low_shelf', 'first_band', 'high_shelf']

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

# RV1 Conv Reverb
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

start_time_total = time.time()

def get_audio_paths(path, names):
    
    audio_path = utils.getFilesPath(path, "*.wav")

    song_names = []
    for p in audio_path:
        song_names.append(os.path.dirname(p))
    song_names = set(song_names)
    audio_path_ = []
    for p in song_names:
        for s in names:
            audio_path_.append(os.path.join(p,s + '.wav'))
        audio_path_.sort()  

    audio_path_dict = {key:[] for key in STEMS}

    for path in audio_path_:
        source_str = os.path.basename(path).split('.wav')[0]
        assert source_str in names
        audio_path_dict[source_str.split('_')[0]].append(path)    
        
    return audio_path_, audio_path_dict

    
def smooth_feature(feature_dict_):
    
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

def get_norm_feature(args_):

    path = args_[0]
    i = args_[1]
    effect = args_[2]
    
    source_str = os.path.basename(path).split('.wav')[0].split('_')[0]
    
    print(f'getting {effect} features for {source_str}- stem {i} of {len(audio_path_)-1} {path}')   
    
    fs, audio = load_wav(path)
    assert(fs == SR)
    all_zeros = not np.any(audio)

    if all_zeros == False:

        conversion_scale = 1. / (1. + np.iinfo(audio.dtype).max)
        audio = audio.astype(dtype=np.float32) * conversion_scale
        audio = np.pad(audio, ((FFT_SIZE, FFT_SIZE), (0, 0)), mode='constant')

        max_db = amp_to_db(np.max(np.abs(audio)))

        if max_db > MIN_DB:

            if effect == 'loudness':
            
                meter = pyln.Meter(SR) 
                loudness = meter.integrated_loudness(audio)
                return [loudness]
            
            elif effect == 'eq':
                
                audio = lufs_normalize(audio, SR, LUFS, log=False) 
                audio_spec = compute_stft(audio,
                                 HOP_LENGTH,
                                 FFT_SIZE,
                                 np.sqrt(np.hanning(FFT_SIZE+1)[:-1]))
                audio_spec = np.abs(audio_spec)
                audio_spec_avg = np.mean(audio_spec, axis=(0,1))
                return audio_spec_avg
            
            elif effect == 'panning':
                phi = get_SPS(audio,
                              n_fft=FFT_SIZE,
                              hop_length=HOP_LENGTH,
                              smooth=False,
                              frames=False)
                return(phi[1])
            
            elif effect == 'compression':
                
                
                x = pyln.normalize.peak(audio, COMP_PEAK_NORM)

                peak_std = get_mean_peak(x,
                                          sr=SR,
                                          true_peak=COMP_TRUE_PEAK,
                                          percentile=COMP_PERCENTILE,
                                          n_mels=comp_settings[source_str]['n_mels'])

                if peak_std is not None:
                    return peak_std
                else:
                    return None
                
                
        else:
            print(f'{path} is silence...')
            return None
        
    else:
            
        print(f'{path} is only zeros...')
        return None

# print('Reverb, Panning, EQ, DRC and loudness normalization... ')

def normalize_audio_path(args_):
    
# for i, path in enumerate(audio_path_):

    path = args_[0]
    i = args_[1]
    effect = args_[2]
    
    source_str = os.path.basename(path).split('.wav')[0]
    src = source_str.split('_')[0]
    
    print(f'{src} normalizing {effect} for {source_str} - stem {i} of {len(audio_path_)-1} {path}')

    if src in STEMS:
        
        start_time = time.time()
        
        fs, audio = load_wav(path)
        
        all_zeros = not np.any(audio)
        
        if all_zeros == False:
        
            assert(fs == SR)

            conversion_scale = 1. / (1. + np.iinfo(audio.dtype).max)
            audio = audio.astype(dtype=np.float32) * conversion_scale
            audio_track = np.pad(audio, ((FFT_SIZE, FFT_SIZE), (0, 0)), mode='constant')
            
            assert len(audio_track.shape) == 2  # Always expects two dimensions
            
            if audio_track.shape[1] == 1:    # Converts mono to stereo with repeated channels
                audio_track = np.repeat(audio_track, 2, axis=-1)    
                
            output_audio = audio_track.copy()
            
            max_db = amp_to_db(np.max(np.abs(output_audio)))
            if max_db > MIN_DB:
            
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
                    
                    if src == 'drums':
                        bands = EQ_BANDS_DRUMS
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
                    
                    if src == 'drums':
                        bands = EQ_BANDS_DRUMS
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
            
            print('PREVIOUS STEM IS ONLY ZEROS')
            
            mix_path = path.split(PATH_DATASET)[-1].split(source_str+'.wav')[0]
            mix_path = PATH_DATASET + mix_path
            mix_path = os.path.join(PATH_DATASET, mix_path + 'mixture.wav')
            
            fs_, audio_ = load_wav(mix_path)
            assert(fs == SR)
            conversion_scale = 1. / (1. + np.iinfo(audio_.dtype).max)
            audio_ = audio_.astype(dtype=np.float32) * conversion_scale
            
            output_audio = np.zeros_like(audio_)     
            
        path = os.path.join(os.path.dirname(path), src)
        output_path = path.split(PATH_DATASET)[-1].split('.wav')[0]
        output_path = PATH_DATASET + output_path
        output_path = os.path.join(PATH_DATASET, output_path + '_' + FILE_NAME + '.wav')
        save_wav(output_path, fs, output_audio, subtype=SUBTYPE)
        print(f'saving {output_path}')
        if all_zeros == False:
            t = (time.time() - start_time)
            print(f' stem {i} of {len(audio_path_)-1} --- took {int(t)} seconds ---')
            tt = (int(t)*(len(audio_path_)-i-1)/3600)/(CPU_COUNT*len(EFFECTS))
            tt = '{:.2f}'.format(tt)
            print(f'--- Finishing {effect} in {tt} hours ---')

audio_path_, audio_path_dict = get_audio_paths(PATH_DATASET, STEMS)


print(f'--- Processing {len(audio_path_)} --- total files')
for p in audio_path_dict:
    print(f'--- Processing {len(audio_path_dict[p])} --- {p} files')

stems_names_ = []
for i,j in enumerate(STEMS):
    stems_names_.append(j+'_'+FILE_NAME)
    
if LOAD_MEAN:
    
    features_mean = np.load(os.path.join(PATH_FEATURES, NAME_FEATURES+'.npy'), allow_pickle='TRUE')[()]
    features_mean = smooth_feature(features_mean)

else:
    
    features_dict = {}
    features_mean = {}
    for effect in EFFECTS:
        features_dict[effect] = {key:[] for key in STEMS}
        features_mean[effect] = {key:[] for key in STEMS}        

stems_names = STEMS.copy()        
for effect in EFFECTS:
    print(f'{effect} ...')
    j=0
    for key in STEMS:
        print(f'{key} ...')
        i = []
        for i_, p_ in enumerate(audio_path_dict[key]):
            i.append(i_)  
        i = np.asarray(i) + j
        j += len(i)

        _func_args = list(zip(audio_path_dict[key], i, [effect]*len(i)))
        
        if LOAD_MEAN is False:
            
            pool = Pool(CPU_COUNT)
            features = pool.map(get_norm_feature, _func_args)
            features_= []
            for j,i in enumerate(features):

                if i is not None:
                    features_.append(i)
            
            features_dict[effect][key] = features_
            
            print(effect, key, len(features_dict[effect][key]))
            s = np.asarray(features_dict[effect][key])
            s = np.mean(s, axis=0)
            features_mean[effect][key] = s
            
            if effect == 'eq':
                assert len(s)==1+FFT_SIZE//2, len(s)
            elif effect == 'compression':
                assert len(s)==2, len(s)
            elif effect == 'panning':
                assert len(s)==1+FFT_SIZE//2, len(s)
            elif effect == 'loudness':
                assert len(s)==1, len(s)
            
            
            if effect == 'eq':
                if key in ['other', 'vocals']:
                    f = 401
                else:
                    f = 151
                features_mean[effect][key] = scipy.signal.savgol_filter(features_mean[effect][key],
                                                                        f, 1, mode='mirror')
            elif effect == 'panning':
                features_mean[effect][key] = scipy.signal.savgol_filter(features_mean[effect][key],
                                                                        501, 1, mode='mirror')
                
        if NORM_AUDIO:        
            pool = Pool(CPU_COUNT)
            pool.map(normalize_audio_path, _func_args)
        
    stems_names = stems_names_
    audio_path_, audio_path_dict = get_audio_paths(PATH_DATASET, stems_names)


# print(features_mean['loudness'])
# print(features_mean['compression'])

if LOAD_MEAN is False:
    np.save(os.path.join(PATH_FEATURES, NAME_FEATURES), features_mean)

t = (time.time() - start_time_total)/3600
t = '{:.2f}'.format(t)
print(f'--- FINISHED - It took {t} hours ---')


            
