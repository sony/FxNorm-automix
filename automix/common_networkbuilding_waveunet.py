"""
Define the structure of the WAVE-U-NET network as introduced by .

Martínez Ramírez M. A., Stoller D. and Moffat D., “A deep learning approach to intelligent drum mixing with the Wave-U-Net” Journal of the Audio Engineering Society, vol. 69, no. 3, pp. 142-151, March 2021

AI Music Technology Group, Sony Group Corporation
AI Speech and Sound Group, Sony Europe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np

from automix.common_datatypes import DataType
from automix.common_supernet import SeparationNet
from automix.common_miscellaneous import pad_to_shape

from typing import Optional



def valid_length(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Return the nearest valid upper length to use with the model so that there is no time steps left over in a 1DConv.

    For all layers, size of the (input - kernel_size) % stride = 0.
    Here valid means that there is no left over frame neglected and discarded.

    Args:
        input_size: size of input
        kernel_size: size of kernel
        stride: stride
        padding: padding
        dilation: dilation

    Returns:
        valid length for convolution
    """
    length = (
        math.ceil(
            (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
        )
        + 1
    )
    length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

    return int(length)


def centre_crop(x, target):
    """
    Center-crop 3-dim. input tensor along last axis so it fits the target tensor shape
    :param x: Input tensor
    :param target: Shape of this tensor will be used as target shape
    :return: Cropped input tensor
    """
    if x is None:
        return None
    if target is None:
        return x

    target_shape = target.shape
    diff = x.shape[-1] - target_shape[-1]
    crop = diff // 2

    if crop == 0:
        return x
    if crop < 0:
        raise ArithmeticError

    return x[:, :, crop : -(diff - crop)].contiguous()






""" Code below taken from original Wave-U-Net PyTorch implementation: https://github.com/f90/Wave-U-Net-Pytorch"""


class UpsamplingBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_shortcut,
        n_outputs,
        kernel_size,
        stride,
        depth,
        conv_type,
        res,
    ):
        super(UpsamplingBlock, self).__init__()
#         assert stride > 1

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(
                n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True
            )

        self.pre_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(
                torch.cat([combined, centre_crop(upsampled, combined)], dim=1)
            )
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size


class DownsamplingBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_shortcut,
        n_outputs,
        kernel_size,
        stride,
        depth,
        conv_type,
        res,
    ):
        super(DownsamplingBlock, self).__init__()
#         assert stride > 1

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

        self.post_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(
                n_outputs, 15, stride
            )  # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(
                n_outputs, n_outputs, kernel_size, stride, conv_type
            )

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size
    

class Waveunet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        num_outputs,
        instruments,
        kernel_size,
        target_output_size,
        conv_type,
        res,
        separate=False,
        depth=1,
        strides=2,
    ):
        super(Waveunet, self).__init__()

        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.depth = depth
        self.instruments = instruments
        self.separate = separate

        # Only odd filter kernels allowed
        assert kernel_size % 2 == 1

        self.waveunets = nn.ModuleDict()

        model_list = instruments if separate else ["ALL"]
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list:
            module = nn.Module()

            module.downsampling_blocks = nn.ModuleList()
            module.upsampling_blocks = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(
                        in_ch,
                        num_channels[i],
                        num_channels[i + 1],
                        kernel_size,
                        strides,
                        depth,
                        conv_type,
                        res,
                    )
                )

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(
                        num_channels[-1 - i],
                        num_channels[-2 - i],
                        num_channels[-2 - i],
                        kernel_size,
                        strides,
                        depth,
                        conv_type,
                        res,
                    )
                )

            module.bottlenecks = nn.ModuleList(
                [
                    ConvLayer(
                        num_channels[-1], num_channels[-1], kernel_size, 1, conv_type
                    )
                    for _ in range(depth)
                ]
            )

            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print(
            "Using valid convolutions with "
            + str(self.input_size)
            + " inputs and "
            + str(self.output_size)
            + " outputs"
        )

        assert (self.input_size - self.output_size) % 2 == 0
        self.shapes = {
            "output_start_frame": (self.input_size - self.output_size) // 2,
            "output_end_frame": (self.input_size - self.output_size) // 2
            + self.output_size,
            "output_frames": self.output_size,
            "input_frames": self.input_size,
        }

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert output_size >= target_output_size
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module):
        """
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        """
        shortcuts = []
        out = x
        # DOWNSAMPLING BLOCKS
        for block in module.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        # OUTPUT CONV
        out = module.output_conv(out)

        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)

        return out

    def forward(self, x, inst=None):
        curr_input_size = x.shape[-1]

        assert (
            curr_input_size == self.input_size
        )  # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            return {inst: self.forward_module(x, self.waveunets[inst])}
        else:
            assert len(self.waveunets) == 1
            out = self.forward_module(x, self.waveunets["ALL"])
            
            return out


class ConvLayer(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, conv_type, transpose=False
    ):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8

        if self.transpose:
            self.filter = nn.ConvTranspose1d(
                n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size - 1
            )
        else:
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        if conv_type == "gn":
            assert n_outputs % NORM_CHANNELS == 0
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)
        # Add you own types of variations here!

    def forward(self, x):
        # Apply the convolution
        if self.conv_type == "gn" or self.conv_type == "bn":
            out = F.relu(self.norm((self.filter(x))))
        else:  # Add your own variations here with elifs conditioned on "conv_type" parameter!
            assert self.conv_type == "normal"
            out = F.leaky_relu(self.filter(x))
        return out

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose:
            curr_size = (
                output_size - 1
            ) * self.stride + 1  # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        curr_size = curr_size + self.kernel_size - 1  # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert (
                curr_size - 1
            ) % self.stride == 0  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert curr_size > 0
        return curr_size

    def get_output_size(self, input_size):
        # Transposed
        if self.transpose:
            assert input_size > 1
            curr_size = (
                input_size - 1
            ) * self.stride + 1  # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = input_size

        # Conv
        curr_size = curr_size - self.kernel_size + 1  # o = i + p - k + 1
        assert curr_size > 0

        # Strided conv/decimation
        if not self.transpose:
            assert (
                curr_size - 1
            ) % self.stride == 0  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1

        return curr_size


class Resample1d(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride,
        transpose=False,
        padding="reflect",
        trainable=False,
    ):
        """
        Creates a resampling layer for time series data (using 1D convolution) - (N, C, W) input format
        :param channels: Number of features C at each time-step
        :param kernel_size: Width of sinc-based lowpass-filter (>= 15 recommended for good filtering performance)
        :param stride: Resampling factor (integer)
        :param transpose: False for down-, true for upsampling
        :param padding: Either "reflect" to pad or "valid" to not pad
        :param trainable: Optionally activate this to train the lowpass-filter, starting from the sinc initialisation
        """
        super(Resample1d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert kernel_size > 2
        assert (kernel_size - 1) % 2 == 0
        assert padding == "reflect" or padding == "valid"

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(
            torch.from_numpy(
                np.repeat(np.reshape(filter, [1, 1, kernel_size]), channels, axis=0)
            ),
            requires_grad=trainable,
        )

    def forward(self, x):
        # Pad here if not using transposed conv
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad = (self.kernel_size - 1) // 2
            out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        # Lowpass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = (input_size - 1) * self.stride + 1
            if self.padding == "valid":
                expected_steps = expected_steps - self.kernel_size + 1

            out = F.conv_transpose1d(
                out, self.filter, stride=self.stride, padding=0, groups=self.channels
            )
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert diff_steps % 2 == 0
                out = out[:, :, diff_steps // 2 : -diff_steps // 2]
        else:
#             assert input_size % self.stride == 1
            out = F.conv1d(
                out, self.filter, stride=self.stride, padding=0, groups=self.channels
            )

        return out

    def get_output_size(self, input_size):
        """
        Returns the output dimensionality (number of timesteps) for a given input size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        """
        assert input_size > 1
        if self.transpose:
            if self.padding == "valid":
                return ((input_size - 1) * self.stride + 1) - self.kernel_size + 1
            else:
                return (input_size - 1) * self.stride + 1
        else:
            assert input_size % self.stride == 1  # Want to take first and last sample
            if self.padding == "valid":
                return input_size - self.kernel_size + 1
            else:
                return input_size

    def get_input_size(self, output_size):
        """
        Returns the input dimensionality (number of timesteps) for a given output size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        """

        # Strided conv/decimation
        if not self.transpose:
            curr_size = (
                output_size - 1
            ) * self.stride + 1  # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        if self.padding == "valid":
            curr_size = curr_size + self.kernel_size - 1  # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert (
                curr_size - 1
            ) % self.stride == 0  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert curr_size > 0
        return curr_size


def build_sinc_filter(kernel_size, cutoff):
    # FOLLOWING https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
    # Sinc lowpass filter
    # Build sinc kernel
    assert kernel_size % 2 == 1
    M = kernel_size - 1
    filter = np.zeros(kernel_size, dtype=np.float32)
    for i in range(kernel_size):
        if i == M // 2:
            filter[i] = 2 * np.pi * cutoff
        else:
            filter[i] = (np.sin(2 * np.pi * cutoff * (i - M // 2)) / (i - M // 2)) * (
                0.42 - 0.5 * np.cos((2 * np.pi * i) / M) + 0.08 * np.cos(4 * np.pi * M)
            )

    filter = filter / np.sum(filter)
    return filter



class Net(SeparationNet):
    """
    WAVEUNET
    """

    input_type = DataType.TIME_SAMPLES
    output_type = DataType.TIME_SAMPLES
    joint_model = True

    def __init__(
        self,
        n_channels, n_targets, n_stems,
        N_FEATURES_ENCODER: Optional[int] = 32,
        LEVELS: Optional[int] = 6,  # original: 6
        OUTPUT_SEQ_LENGTH: Optional[float] = 44100,
        FEATURE_GROWTH: Optional[str] = "double",
        STRIDES: Optional[int] = 1,
        KERNEL_SIZE_ENCODER: Optional[int] = 5,
        CONV_TYPE: Optional[str] = "gn",
        RES: Optional[str] = "fixed",
        DEPTH: Optional[int] = 1,
        **kwargs
    ):

        """
        Initialize the network.

        Args:
            input_offset: bias to apply to the input
            input_scale: scale to apply to the input
            output_offset: bias to apply to the output
            output_scale: scale to apply to the output
            n_channels: number of channels
            n_targets: number of targets
            N_FEATURES_ENCODER: number of encoder features
            KERNEL_SIZE_ENCODER: encoder kernel size
            N_FEATURES_SEPARATION_MODULE: number of separation module features
            N_FEATURES_OUT: number of output features
            N_FEATURES_TB: number of features in temporal blocks
            KERNEL_SIZE_TB: kernel size in temporal blocks
            N_TB_PER_REPEAT: number of temporal blocks per repeat
            N_REPEATS: number of repeats
            kwargs: additional arguments
        """
        super(Net, self).__init__()
        
        self.n_targets = n_targets
        self.n_stems = n_stems
        
        self.channels = n_channels
        self.features = N_FEATURES_ENCODER
        self.levels = LEVELS
        self.feature_growth = FEATURE_GROWTH
        self.output_size = OUTPUT_SEQ_LENGTH
        self.kernel_size = KERNEL_SIZE_ENCODER
        self.conv_type = CONV_TYPE
        self.res = RES
        self.depth = DEPTH
        self.strides = STRIDES

        self.num_features = (
            [self.features * i for i in range(1, self.levels + 1)]
            if self.feature_growth == "add"
            else [self.features * 2 ** i for i in range(0, self.levels)]
        )

        target_outputs = self.output_size

        self.network = Waveunet(
            num_inputs=self.channels*self.n_stems,
            num_channels=self.num_features,
            num_outputs=self.channels*self.n_targets,
            instruments=["mixture"],
            kernel_size=self.kernel_size,
            target_output_size=target_outputs,
            conv_type=self.conv_type,
            res=self.res,
            separate=False,
            depth=self.depth,
            strides=self.strides,
        )


    def forward(self, x):
        out = self.network(x)
        out2 = out.clone()
        out2 = out2.unsqueeze(0)
        return out2

        
    def reset_parameters(self, heuristic='init_chrono'):
        pass

                            
 