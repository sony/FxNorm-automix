"""
Loss functions for training separation networks.

The following tensor layout convention is used:
* Spectral loss: output/target tensors have shape ...xFxN
* Time-domain loss: output/target tensors have shape ...xT

AI Music Technology Group, Sony Group Corporation
AI Speech and Sound Group, Sony Europe
"""
from itertools import permutations
from numpy import delete, arange
import torch
import torch.nn as nn
from typing import List, Optional

import numpy as np
import scipy

from automix.common_datatypes import DataType, time_domain_to_spectral, remove_guards


class Loss(nn.Module):
    """
    Generic class for computing a training/validation loss.

    The required data type for `out` and `tar` is given by `self.data_type`
    which can be either just an element from `DataType` or a set/list of
    `DataType` elements (e.g., for `LinearCombinationLoss`)

    By setting `guard_left`/`guard_right` to a non-zero value, we can ignore
    samples (in the data type `self.data_type`) in `out` and `tar`. This is useful
    to avoid possible issues at the boundaries (e.g., due to padding inside a CNN).
    """

    def __init__(self, loss: nn.Module, data_type: DataType, scale: float = 1.,
                 guard_left: int = 0, guard_right: int = 0):
        """
        Initialize the loss.

        Args:
            loss: the loss module
            data_type: the datatype of the variables the loss is computed upon
            scale: scalar factor for the loss value
            guard_left: number of frames to ignore in loss computation on the left
            guard_right: number of frames to ignore in loss computation on the right
        """
        super().__init__()

        # register loss with its data type
        self._loss = loss
        self.data_type = data_type

        # add guard
        assert guard_left >= 0  # we shouldn't allow negative slicing
        assert guard_right >= 0  # we shouldn't allow negative slicing
        self.guard_left = guard_left
        self.guard_right = guard_right

        # register scaling coefficient (useful, e.g., for time-domain losses)
        self.register_buffer('_scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, out: dict, tar: dict):
        """
        Forward pass.

        Compute loss by using correct data type from `out` and `tar`.

        Args:
            out: output of the networks
            tar: ground truth target

        Returns:
             loss value
        """
        return self._loss(
            remove_guards(out[self.data_type], self.guard_left, self.guard_right, self.data_type),
            remove_guards(tar[self.data_type], self.guard_left, self.guard_right, self.data_type)
        ).mul_(self._scale)



    

    
    
    
    
class FIRFilterLoss(Loss):

    def __init__(self, loss: Loss, filter_type: List[str],
                 coef=0.85, fs=44100, ntaps=101, amp=True,
                 fft_size=4096, hop_length=1024, stft_window=np.hanning(4096+1)):
        """
        Initialize the loss. FIRFIlters are applied in series in the time domain.

        Args:
             filter_type lsit(str): List of the desired FIR filters ("hp", "fd", "aw", "lp"). 
             coef (float): Coefficient value for the filter tap. Default: 0.85
             ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
             
             if ._loss.data_type is in the Frequency domain:
             
             fft_size (int): size of FFT
             hop_length (int): hop length
             stft_window (numpy): window function for STFT
        """
        super().__init__(loss, DataType.TIME_SAMPLES)
        
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.amp = amp
        
        self.filter_type = filter_type
        self.filters = []
        self.weights = []
        for fir_type in self.filter_type:
            self.weights.append(self.get_weights(fir_type,
                                                self.coef,
                                                self.fs, 
                                                self.ntaps))
        for i in range(len(self.filter_type)):
            self.register_buffer(f'_weights{i}', self.weights[i])
            
        self._fft_size = fft_size
        self._hop_length = hop_length
        self.register_buffer('_window', torch.from_numpy(stft_window))            
        

    def get_weights(self, filter_type, coef, fs, ntaps):
        
        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            fir.weight.requires_grad = False
            fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        if filter_type == "lp":
            fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            fir.weight.requires_grad = False
            fir.weight.data = torch.tensor([1, coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            fir.weight.requires_grad = False
            fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                 [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                 [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
             )
            DENs = np.polymul(
                 np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
             )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            fir = torch.nn.Conv1d(
                 1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
             )
            fir.weight.requires_grad = False
            fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

        if self.amp:    
            fir.weight.data = fir.weight.data.type(torch.HalfTensor)

        return fir.weight.data

    def process_filter(self, inp, weights):
        """Calculate filter output.
        Args:
             inp (Tensor): signal (sources, B, #channels, #samples).
        Returns:
             Tensor: Filtered signal.
        """

        # get number of sources in minibatch
        # inp = inp_.clone()
        n_sources = inp.shape[0]
        n_channels = inp.shape[2]
        n_samples = inp.shape[-1]
        for i in range(n_sources):
            ch = []
            for j in range(n_channels):
                ch_ = torch.unsqueeze(inp[i,:,j,:], 1)
                ch__ = torch.nn.functional.conv1d(ch_.clone(),
                                                weights,
                                                padding=self.ntaps // 2)
                ch.append(ch__) 
            ch = torch.hstack(ch)
            inp[i, ...] = ch[..., :n_samples] 
        return inp
    
    def forward(self, out: dict, tar: dict):
        """
        Forward pass.

        Compute loss by using correct data type from `out` and `tar`.

        Args:
             out: output of the networks
             tar: ground truth target

        Returns:
              loss value
         """
        
        for i in range(len(self.filter_type)):

            out[self.data_type] = self.process_filter(out[self.data_type],
                                                            self.__getattr__(f'_weights{i}'))
            tar[self.data_type] = self.process_filter(tar[self.data_type],
                                                            self.__getattr__(f'_weights{i}'))
            
        if self._loss.data_type in [DataType.STFT_MAGNITUDE,
                                    DataType.STFT_MAGNITUDEPHASE,
                                    DataType.STFT_COMPLEX]:
        
            out[self._loss.data_type] = time_domain_to_spectral(out[self.data_type], self._loss.data_type,
                                                            self._window, self._fft_size, self._hop_length)

            tar[self._loss.data_type] = time_domain_to_spectral(tar[self.data_type], self._loss.data_type,
                                                            self._window, self._fft_size, self._hop_length)

        

        return self._loss(out, tar)

    
    
class StereoLoss(Loss):
    """
    Stereo Loss function based on https://arxiv.org/pdf/2010.10291.pdf
    StereoLoss = "La"
    """

    def __init__(self, loss: Loss, w_sc=1.0,
                 w_smag=1.0, log=True, filter_type=[],
                 coef=0.85, fs=44100, ntaps=101, amp=True,
                 fft_size=4096, hop_length=1024, stft_window=np.sqrt(np.hanning(4096+1)[:-1])):
        """
        Initialize the Stereo Loss, includes FIR filter in the time domain and Spectral Convergence
        and LogMagnitude https://arxiv.org/pdf/2010.10291.pdf

        Args:
             filter_type lsit(str): List of the desired FIR filters ("hp", "fd", "aw", "lp"). 
             coef (float): Coefficient value for the filter tap. Default: 0.85
             ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
             fft_size (int): size of FFT
             hop_length (int): hop length
             stft_window (numpy): window function for STFT
        """
        super().__init__(loss, loss.data_type)
        
        self.filter_type = filter_type
        
        
        if self.filter_type:
            self.fir = FIRFilterLoss(loss,
                                     filter_type=filter_type,
                                     coef=coef, fs=fs, ntaps=ntaps, amp=amp)
        
        self._fft_size = fft_size
        self._hop_length = hop_length
        self.register_buffer('_window', torch.from_numpy(stft_window))      
        
        self.w_sc = w_sc
        self.w_smag = w_smag
        self.log = log
        
        
    def sum_diff(self, x):
        """
        Computes sum and diff of stereo channells

        Args:
             x: input tensor (1, batch, channels, samples)

        Returns:
              sum and diff tensors (1, batch, 1, samples)
         """
        
        
        channels = torch.unbind(x, dim=2)

        sum_ = torch.stack(channels, dim=0).sum(dim=0, keepdim=False)
        sum_ = sum_.unsqueeze(dim=2)
        
        diff_ = torch.stack([channels[0],-1.0*channels[1]], dim=0).sum(dim=0, keepdim=False)
        diff_ = diff_.unsqueeze(dim=2)
        
        return sum_, diff_
        
    def spectral_convergence_loss(self, x, y):
        """
        Computes spectral convergence loss

        Args:
             x: output spectral tensor 
             y: target spectral tensor 
        Returns:
              Spectral Convergence loss
         """

        return torch.norm(y - x, p="fro") / torch.norm(y, p="fro")
    
    def compute_spectral(self, x, datatype, win, fft, hop):
        """
        Computes spectral represenation and applies guards

        Args:
             x: tensor of TIME_SAMPLES
        Returns:
              Spectral magnitude tensor
         """
        
        X = time_domain_to_spectral(x,
                                    datatype,
                                    win,
                                    fft,
                                    hop)
        
        X = remove_guards(X, self._loss.guard_left, self._loss.guard_right, datatype)
  
        return X
    
    def spectral_magnitude_loss(self, x, y):
        """
        Computes spectral log_magnitude loss

        Args:
             x: output tensor of STFT_MAGNITUDE
             y: target tensor of STFT_MAGNITUDE
        Returns:
              Spectral log magnitude loss
         """
        
        if self.log:
            x = torch.log(1e-30 + x)
            y = torch.log(1e-30 + y)
   
        return self._loss._loss(x, y)
    
    def forward(self, out: dict, tar: dict):
        """
        Forward pass.

        Compute loss by using correct data type from `out` and `tar`.

        Args:
             out: output of the networks
             tar: ground truth target

        Returns:
              loss value
         """
   
        out_ = out[DataType.TIME_SAMPLES]
        tar_ = tar[DataType.TIME_SAMPLES]
    
        if self.filter_type:
            for i in range(len(self.filter_type)):

                out_ = self.fir.process_filter(out_,
                                               self.fir.__getattr__(f'_weights{i}'))
                tar_ = self.fir.process_filter(tar_,
                                               self.fir.__getattr__(f'_weights{i}'))
                
        out_sum, out_diff = self.sum_diff(out_)
        tar_sum, tar_diff = self.sum_diff(tar_)
    
        out_sum_mag = self.compute_spectral(out_sum,
                                            self._loss.data_type,
                                            self._window,
                                            self._fft_size,
                                            self._hop_length)

        out_diff_mag = self.compute_spectral(out_diff,
                                             self._loss.data_type,
                                             self._window,
                                             self._fft_size,
                                             self._hop_length)
        
        tar_sum_mag = self.compute_spectral(tar_sum,
                                            self._loss.data_type,
                                            self._window,
                                            self._fft_size,
                                            self._hop_length)

        tar_diff_mag = self.compute_spectral(tar_diff,
                                             self._loss.data_type,
                                             self._window,
                                             self._fft_size,
                                             self._hop_length)
        
        sc_sum = self.spectral_convergence_loss(out_sum_mag, tar_sum_mag)
        sc_diff = self.spectral_convergence_loss(out_diff_mag, tar_diff_mag)    
        
        sm_sum = self.spectral_magnitude_loss(out_sum_mag, tar_sum_mag)
        sm_diff = self.spectral_magnitude_loss(out_diff_mag, tar_diff_mag)
        
        loss_sum = self.w_sc*(sc_sum) + self.w_smag*(sm_sum)
        loss_diff = self.w_sc*(sc_diff) + self.w_smag*(sm_diff)
       
        return loss_sum+loss_diff 
    
    
class StereoLoss2(Loss):
    """
    Stereo Loss function based on https://arxiv.org/pdf/2010.10291.pdf
    StereoLoss2 = "Lb"
    """

    def __init__(self, loss: Loss, w_sc=1.0,
                 w_smag=1.0, log=True, filter_type=[],
                 coef=0.85, fs=44100, ntaps=101, amp=True,
                 fft_size=4096, hop_length=1024, stft_window=np.sqrt(np.hanning(4096+1)[:-1])):
        """
        Initialize the Stereo Loss, includes FIR filter in the time domain and Spectral Convergence
        and LogMagnitude https://arxiv.org/pdf/2010.10291.pdf

        Args:
             filter_type lsit(str): List of the desired FIR filters ("hp", "fd", "aw", "lp"). 
             coef (float): Coefficient value for the filter tap. Default: 0.85
             ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
             fft_size (int): size of FFT
             hop_length (int): hop length
             stft_window (numpy): window function for STFT
        """
        super().__init__(loss, loss.data_type)
        
        self.filter_type = filter_type
        
        
        if self.filter_type:
            self.fir = FIRFilterLoss(loss,
                                     filter_type=filter_type,
                                     coef=coef, fs=fs, ntaps=ntaps, amp=amp)
        
        self._fft_size = fft_size
        self._hop_length = hop_length
        self.register_buffer('_window', torch.from_numpy(stft_window))      
        
        self.w_sc = w_sc
        self.w_smag = w_smag
        self.log = log
        
        self.lossMSE = nn.MSELoss() 
        
        
        
    def sum_diff(self, x):
        """
        Computes sum and diff of stereo channells

        Args:
             x: input tensor (1, batch, channels, samples)

        Returns:
              sum and diff tensors (1, batch, 1, samples)
         """
        
        
        channels = torch.unbind(x, dim=2)

        sum_ = torch.stack(channels, dim=0).sum(dim=0, keepdim=False)
        sum_ = sum_.unsqueeze(dim=2)
        
        diff_ = torch.stack([channels[0],-1.0*channels[1]], dim=0).sum(dim=0, keepdim=False)
        diff_ = diff_.unsqueeze(dim=2)
        
        return sum_, diff_
        
    def spectral_convergence_loss(self, x, y):
        """
        Computes spectral convergence loss

        Args:
             x: output spectral tensor 
             y: target spectral tensor 
        Returns:
              Spectral Convergence loss
         """

        return self.lossMSE(x,y)
    
    def compute_spectral(self, x, datatype, win, fft, hop):
        """
        Computes spectral represenation and applies guards

        Args:
             x: tensor of TIME_SAMPLES
        Returns:
              Spectral magnitude tensor
         """
        
        X = time_domain_to_spectral(x,
                                    datatype,
                                    win,
                                    fft,
                                    hop)
        
        X = remove_guards(X, self._loss.guard_left, self._loss.guard_right, datatype)
  
        return X
    
    def spectral_magnitude_loss(self, x, y):
        """
        Computes spectral log_magnitude loss

        Args:
             x: output tensor of STFT_MAGNITUDE
             y: target tensor of STFT_MAGNITUDE
        Returns:
              Spectral log magnitude loss
         """
        
        if self.log:
            x = torch.log(1e-30 + x)
            y = torch.log(1e-30 + y)
   
        return self._loss._loss(x, y)
    
    def forward(self, out: dict, tar: dict):
        """
        Forward pass.

        Compute loss by using correct data type from `out` and `tar`.

        Args:
             out: output of the networks
             tar: ground truth target

        Returns:
              loss value
         """
   
        out_ = out[DataType.TIME_SAMPLES]
        tar_ = tar[DataType.TIME_SAMPLES]
    
        if self.filter_type:
            for i in range(len(self.filter_type)):

                out_ = self.fir.process_filter(out_,
                                               self.fir.__getattr__(f'_weights{i}'))
                tar_ = self.fir.process_filter(tar_,
                                               self.fir.__getattr__(f'_weights{i}'))
                
        out_sum, out_diff = self.sum_diff(out_)
        tar_sum, tar_diff = self.sum_diff(tar_)
    
        out_sum_mag = self.compute_spectral(out_sum,
                                            self._loss.data_type,
                                            self._window,
                                            self._fft_size,
                                            self._hop_length)

        out_diff_mag = self.compute_spectral(out_diff,
                                             self._loss.data_type,
                                             self._window,
                                             self._fft_size,
                                             self._hop_length)
        
        tar_sum_mag = self.compute_spectral(tar_sum,
                                            self._loss.data_type,
                                            self._window,
                                            self._fft_size,
                                            self._hop_length)

        tar_diff_mag = self.compute_spectral(tar_diff,
                                             self._loss.data_type,
                                             self._window,
                                             self._fft_size,
                                             self._hop_length)
        
        sc_sum = self.spectral_convergence_loss(out_sum_mag, tar_sum_mag)
        sc_diff = self.spectral_convergence_loss(out_diff_mag, tar_diff_mag)    
        
        sm_sum = self.spectral_magnitude_loss(out_sum_mag, tar_sum_mag)
        sm_diff = self.spectral_magnitude_loss(out_diff_mag, tar_diff_mag)
        
        loss_sum = self.w_sc*(sc_sum) + self.w_smag*(sm_sum)
        loss_diff = self.w_sc*(sc_diff) + self.w_smag*(sm_diff)
       
        return loss_sum+loss_diff 
    
    
