import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import math
import fnmatch
import os

def getFilesPath(directory, extension):
    
    n_path=[]
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, extension):
                n_path.append(os.path.join(path,name))
    n_path.sort()
                
    return n_path




