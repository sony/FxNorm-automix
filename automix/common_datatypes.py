"""
Common data conversion functions.

This module contains common conversion functions, such as from time-domain to STFT domain.

AI Music Technology Group, Sony Group Corporation
AI Speech and Sound Group, Sony Europe
"""
import torch
from enum import IntEnum
from automix.common_miscellaneous import valid_length


class DataType(IntEnum):
    """Data representation types for separation network."""

    # STFT magnitude with shape
    #  inp:             [batch_size x n_channels x n_freqbins x n_frames]
    #  out: [n_targets x batch_size x n_channels x n_freqbins x n_frames]
    STFT_MAGNITUDE = 1

    # STFT magnitude and phase with shape
    #  inp:             [batch_size x n_channels x n_freqbins x n_frames x 2]
    #  out: [n_targets x batch_size x n_channels x n_freqbins x n_frames x 2]
    STFT_MAGNITUDEPHASE = 2

    # STFT real/imaginary part with shape
    #  inp:             [batch_size x n_channels x n_freqbins x n_frames x 2]
    #  out: [n_targets x batch_size x n_channels x n_freqbins x n_frames x 2]
    STFT_COMPLEX = 3

    # Time samples with shape
    #  inp:             [batch_size x n_channels x n_samples]
    #  out: [n_targets x batch_size x n_channels x n_samples]
    TIME_SAMPLES = 4


def mp2ri(spectrogram):
    """
    Convert magnitude/phase spectrogram into real/imaginary spectrogram.

    It is assumed that element `0` in the last dimension of spectrogram is the magnitude and element `1` is the phase.

    Args:
        spectrogram: input spectrogram as [magnitude, phase]

    Returns:
        Complex spectrogram as [real_part, imag_part]
    """
    r = torch.mul(spectrogram[..., 0], torch.cos(spectrogram[..., 1]))
    i = torch.mul(spectrogram[..., 0], torch.sin(spectrogram[..., 1]))
    return torch.stack([r, i], dim=-1)


def ri2mp(spectrogram):
    """
    Convert a real/imaginary spectrogram into magnitude/phase spectrogram.

    It is assumed that element `0` in the last dimension of spectrogram is the real part and element `1` is
    the imaginary part.

    Args:
        spectrogram: input spectrogram as [real_part, imag_part]

    Returns:
        Complex spectrogram as [magnitude, phase]
    """
    m = torch.norm(spectrogram, p=2, dim=-1)
    p = torch.atan2(spectrogram[..., 1], spectrogram[..., 0])
    return torch.stack([m, p], dim=-1)


def ri2m(spectrogram):
    """
    Compute magnitude spectrogram from real/imaginary spectrogram.

    It is assumed that element `0` in the last dimension of spectrogram is the real part and element `1` is the
    imaginary part.

    Args:
        spectrogram: input spectrogram as [real_part, imag_part]

    Returns:
        Magnitude spectrogram
    """
    return torch.norm(spectrogram, p=2, dim=-1)


def spectral_to_time_domain(spectrogram, spectrogram_type, window, fft_size, hop_length):
    """
    Convert STFT (either given by real/imaginary part or magnitude/phase) into the time-domain.

    Note: `torch.istft` is called with `center=True` to allow the inversion of the STFT operation.
    We lose `fft_size - hop_length` samples at the begin/end, i.e.,

    ```
    trim = fft_size - hop_length
    x[trim:-trim] == spectral_to_time_domain(time_domain_to_spectral(x))
    ```

    Args:
        spectrogram: real/imaginary or magnitude/phase spectrogram, shape ...xFxNx2
        spectrogram_type: format of provided spectrogram
        window: window function
        fft_size: size of FFT
        hop_length: hop length

    Returns:
        time_signal: shape ...xT
    """
    assert(spectrogram_type in [DataType.STFT_MAGNITUDEPHASE,
                                DataType.STFT_COMPLEX])

    if spectrogram_type == DataType.STFT_MAGNITUDEPHASE:
        spectrogram = mp2ri(spectrogram)

    shp = spectrogram.shape
    time_signal = torch.istft(spectrogram.view(-1, *shp[-3:]),
                              n_fft=fft_size, hop_length=hop_length,
                              window=window, center=True,
                              return_complex=False)  # time_signal: ...xT
    time_signal = time_signal.view(shp[:-3] + time_signal.shape[-1:])

    # We return only the inner part of the signal to avoid boundary effects of the
    # STFT computation.
    # iSTFT internally removes `fft_size//2` if we use `center=True`.
    # By using larger overlap we still need to remove `fft_size-hop_length - fft_size // 2`.
    trim_amount = fft_size - hop_length - fft_size // 2
    if trim_amount > 0:
        return time_signal[..., trim_amount:-trim_amount]
    else:
        return time_signal


def time_domain_to_spectral(time_signal, spectrogram_type, window, fft_size, hop_length):
    """
    Convert time-domain signal into the STFT domain (real/imaginary or magnitude/phase or magnitude only).

    Args:
        time_signal: time-domain signal, shape ...xT
        spectrogram_type: format of output spectrogram
        window: window function
        fft_size: size of FFT
        hop_length: hop length

    Returns:
        spectrogram: shape ...xFxNx2 (real/imag or magn/phase) or ...xFxN (magn only)
    """
    assert(spectrogram_type in [DataType.STFT_MAGNITUDE,
                                DataType.STFT_MAGNITUDEPHASE,
                                DataType.STFT_COMPLEX])

    shp = time_signal.shape
    spectrogram = torch.stft(time_signal.view(-1, shp[-1]), n_fft=fft_size,
                             hop_length=hop_length, window=window, center=False,
                             return_complex=False)

    if spectrogram_type == DataType.STFT_MAGNITUDEPHASE:
        spectrogram = ri2mp(spectrogram).view(shp[:-1] + spectrogram.shape[-3:])
    elif spectrogram_type == DataType.STFT_MAGNITUDE:
        spectrogram = ri2m(spectrogram).view(shp[:-1] + spectrogram.shape[-3:-1])
    elif spectrogram_type == DataType.STFT_COMPLEX:
        spectrogram = spectrogram.view(shp[:-1] + spectrogram.shape[-3:])

    return spectrogram


# INFERENCE WITH BATCHED MIXTURES
def get_length(inp: torch.Tensor, input_type: DataType) -> int:
    """
    Get length from `input` with datatype `input_type` (input).

    Args:
        inp: input data
        input_type: datatype of input

    Returns:
        original length of the input data
    """
    if input_type == DataType.TIME_SAMPLES:
        original_length = inp.shape[2]
    else:
        original_length = inp.shape[3]

    return original_length


def remove_guards(inp: torch.Tensor, guard_left: int, guard_right: int, input_type: DataType) -> torch.Tensor:
    """
    Remove guards.

    Ignore `guard_left`/`guard_right` samples to the left/to the right of `input`
    (taking into account the datatype of `input`).

    Args:
        inp: input data
        guard_left: samples to remove on the left side
        guard_right: samples to remove on the right side
        input_type: datatype of input

    Returns:
        sliced input
    """
    # slicing with 0 may remove the dimension - take care here
    slice_left = None if guard_left == 0 else guard_left
    slice_right = None if guard_right == 0 else -guard_right

    if input_type == DataType.TIME_SAMPLES:
        return inp[:, :, :, slice_left:slice_right]
    elif input_type == DataType.STFT_MAGNITUDE:
        return inp[:, :, :, :, slice_left:slice_right]
    else:
        return inp[:, :, :, :, slice_left:slice_right, ...]


def unfold(inp: torch.Tensor, window_size: int, guard_left: int, guard_right: int,
           input_type: DataType) -> (torch.Tensor, int):
    """
    Unfold a long sequence.

    Transform long sequence `input` with datatype `input_type` and ** with a batch size of 1 ** into a batch of smaller
    segments of the input, that can be reconstructed taking the center portion of each batch.

    This can be useful for cases where inference must be done on short segments to fit the conditions in which the
    training has been performed.

    guard_right may be set to 0 for causal models.

    Args:
        inp: input with dimensions in accordance with DataType
        window_size: length of each window in `input` DataType samples format
        guard_left: values to slice to the left because "contaminated" due to being on the border
        guard_right: values to slice to the right because "contaminated" due to being on the border
        input_type: DataType of `input`

    Returns:
        the input sliced as shorter segments with overlap, each slice being saved into the batch dimension
    """
    assert(inp.shape[0] == 1)

    hop_size = window_size - guard_left - guard_right
    assert hop_size > 0, '`hop_size` for unfold not positive - `guard_left/right` are too large'

    if input_type == DataType.TIME_SAMPLES:
        original_length = inp.shape[2]
        samples_dim = 2
    else:
        original_length = inp.shape[3]
        samples_dim = 3

    # we will pad first on the left `guard_left` zeros, on the right `guard_right`,
    # and complete with `necessary_padding` zeros on the right if necessary,
    # making sure too that there are enough zeros on the right to use all values
    necessary_padding = valid_length(input_size=guard_left + original_length + guard_right,
                                     kernel_size=window_size,
                                     stride=hop_size,
                                     padding=0,
                                     dilation=1) - (guard_left + original_length + guard_right)

    # we pad the sample dimension, which may not be the last dim depending on the DataType
    right_unpadded_dims = len(inp.shape) - samples_dim - 1

    # we need the last convolution to be full
    # TODO: change to reflective padding mode once this is available in PyTorch (see issue #46)
    inp = torch.nn.functional.pad(inp,
                                  pad=((0, 0) * right_unpadded_dims + (guard_left, necessary_padding + guard_right)))

    # out shape 1 x n_channels x ... x n_windows x ... x window_size
    #                                      |                 |
    #                                 samples_dim       new dimension
    batched_output = inp.unfold(dimension=samples_dim, size=window_size, step=hop_size)

    # put n_windows in the batch dimension
    # out shape n_windows x n_channels x ... x 1 x ... x window_size
    batched_output = torch.transpose(batched_output, 0, samples_dim)

    # unfold adds in the last dimension the unfolded values, put it back at the samples dimension
    # out shape n_windows x n_channels x ... x 1
    batched_output = torch.transpose(batched_output, samples_dim, -1)

    # Finally, we remove the last dimension to match DataType dimension
    # out shape n_windows x n_channels x ...
    batched_output = torch.squeeze(batched_output, -1)

    return batched_output


def reconstruct_from_unfold(inp: torch.Tensor, window_size: int, guard_left: int, guard_right: int,
                            original_length: int, input_type: DataType) -> torch.Tensor:
    """
    Reconstruct a tensor after unfolding it.

    Reconstructs a longer sequence from unfolded `input` with datatype `input_type`, that is a batched version of
    the signal of interest as constructed in `unfold` and processed by a model (which will add a new dimension
    for `n_targets`):

    ```
    # unfold into batch dimension
    x_unfolded = unfold(x, ...)

    # apply network
    x_processed = net(x_unfolded)

    # fold back from batch into time/frame dimension
    x_reconstructed = reconstruct_from_unfold(x_processed, ...)
    ```

    guard_right may be set to 0 for causal models.

    The output has a batch size of 1 compared to the input `inp` that had a batch size of n_windows

    Args:
        inp: batched signal with dimensions in accordance with DataType
        window_size: length of each window in `input` DataType samples format
        guard_left: values to slice to the left because "contaminated" due to being on the border
        guard_right: values to slice to the right because "contaminated" due to being on the border.
        original_length: number of values (n_samples or n_frames) of the original input before unfold
        input_type: DataType of `input`

    Returns:
        `output`: reconstructed signal with dimensions in accordance with DataType (output).
    """
    if input_type == DataType.TIME_SAMPLES:
        assert inp.ndim == 4, ('Expecting 4D input of shape '
                               'n_targets x batch_size x n_channels x n_samples')
        samples_dim = 3
    elif input_type == DataType.STFT_MAGNITUDE:
        assert inp.ndim == 5, ('Expecting 5D input of shape '
                               'n_targets x batch_size x n_channels x n_freqbins x n_frames')
        samples_dim = 4
    elif input_type in [DataType.STFT_MAGNITUDEPHASE,
                        DataType.STFT_COMPLEX]:
        assert inp.ndim == 6, ('Expecting 6D input of shape '
                               'n_targets x batch_size x n_channels x n_freqbins x n_frames x 2')
        samples_dim = 4

    # we remove outer regions on the `samples_dim` dimension
    output = inp.narrow(dim=samples_dim, start=guard_left, length=window_size - guard_right - guard_left)

    if input_type == DataType.TIME_SAMPLES:
        # out shape n_targets x n_channels x n_windows x window_size_trimmed
        output = output.permute((0, 2, 1, 3))
    elif input_type == DataType.STFT_MAGNITUDE:
        # out shape n_targets x n_channels x n_freqbins x n_windows x window_size_trimmed
        output = output.permute((0, 2, 3, 1, 4))
    elif input_type in [DataType.STFT_MAGNITUDEPHASE, DataType.STFT_COMPLEX]:
        # out shape n_targets x n_channels x n_freqbins x 2 x n_windows x window_size_trimmed
        output = output.permute((0, 2, 3, 5, 1, 4))

    # we merge the last two axis together, note that this makes the batch axis disappear
    output = torch.reshape(output, (output.shape[:-2] + (-1,)))

    # trim the end, as the front `guard_left` values have already been trimmed
    output = output[..., :original_length]

    # put back the `2` dimension at the end if needed
    if input_type in [DataType.STFT_MAGNITUDEPHASE, DataType.STFT_COMPLEX]:
        output = output.permute((0, 1, 2, 4, 3))

    output = output[:, None, ...]  # add the batch dimension to comply with DataType standards

    return output
