import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import math
import fnmatch
import os
import pyloudnorm


def getFilesPath(directory, extension):
    
    n_path=[]
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, extension):
                n_path.append(os.path.join(path,name))
    n_path.sort()
                
    return n_path


def lufs_normalize(x, sr, lufs, log=True):

    # measure the loudness first 
    meter = pyloudnorm.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(x+1e-10)
    if log:
        print("original loudness: ", loudness," max value: ", np.max(np.abs(x)))

    loudness_normalized_audio = pyloudnorm.normalize.loudness(x, loudness, lufs)
    
    maxabs_amp = np.maximum(1.0, 1e-6 + np.max(np.abs(loudness_normalized_audio)))
    loudness_normalized_audio /= maxabs_amp
    
    loudness = meter.integrated_loudness(loudness_normalized_audio)
    if log:
        print("new loudness: ", loudness," max value: ", np.max(np.abs(loudness_normalized_audio)))

    
    return loudness_normalized_audio

