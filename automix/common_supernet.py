"""
Module that incorporates both the network and the loss functions, useful for parallelization in a multiGPU setting.

AI Music Technology Group, Sony Group Corporation
AI Speech and Sound Group, Sony Europe
"""
import torch
import torch.nn as nn
import torch.cuda.amp as amp

from automix.common_datatypes import DataType
from automix.common_datatypes import ri2m, ri2mp, mp2ri, spectral_to_time_domain, time_domain_to_spectral
from automix.common_datatypes import unfold, reconstruct_from_unfold, get_length
from automix.common_miscellaneous import uprint





class SuperNet(nn.Module):
    """Combine input preprocessing, separation network, output processing and loss function into one module."""

    def __init__(self, net: SeparationNet, stft_window,
                 stft_hop_length, training_batch_size, batched_valid, training_length, unfolding_params=None,
                 use_amp=False):
        """
        Initialize the SuperNet.

        Args:
            net: network module
            stft_window: window for STFT
            stft_hop_length: hop length
            training_batch_size: batch size
            batched_valid: whether to do validation by splitting a long sequence over the batch dimension
            training_length: training sequence length
            unfolding_params: parameters for unfold
            use_amp: use mixed-precision training
        """
        super(SuperNet, self).__init__()

        self.net = net
        assert(self.net.input_type == self.net.output_type)  # assume that input/output type match

        self.register_buffer('_stft_window', stft_window)
        self._stft_fft_size = self._stft_window.shape[0]
        self._stft_hop_length = stft_hop_length
        self.q_op = None
        self.q_bw = None
        self.q_params = {}
        self.unfolding_params = unfolding_params
        self.batched_valid = batched_valid
        self.training_length = training_length
        self.training_batch_size = training_batch_size
        self.batched_forward = lambda x, m: torch.cat([m(_in) for _in in torch.split(x, self.training_batch_size)],
                                                      dim=1)
        self.use_amp = use_amp

    def _iterate_modules_recursive(self, module: nn.Module, apply_fn):
        """
        Iterate over all parameters of a module once, including its submodules.

        Apply the function apply_fn to all parameters during the iteration
        An apply_fn needs to have the signature apply_fn(key: str, p: nn.Parameter, module: nn.Module)

        Args:
            module: PyTorch module
            apply_fn: function to apply to the module's parameters
        """
        # get all parameters for the current module
        named_params = [(key, p) for key, p in module.named_parameters(recurse=False)]

        # apply function to them
        for key, p in named_params:
            apply_fn(key, p, module)

        # recurse through all children
        for child in module.children():
            self._iterate_modules_recursive(child, apply_fn)

    def _insert_soft_weights(self, p_name: str, p: nn.Parameter, module: nn.Module):
        """
        Add the float32 parameters to the module and delete the original ones.

        This is an apply_fn for the method _iterate_modules_recursive.

        Args:
            p_name: parameter name
            p: apply the function to this parameter
            module: the module
        """
        if not hasattr(module, 'q_params'):
            module.q_params = {}

        # compute quantization parameters for this parameter
        # this is currently done once and then fixed throughout the training
        module.q_params[p_name] = self.q_op.compute_parameters(p, self.q_bw)

        # print parameters to log file
        uprint(f'\t{p_name:>30}: {module.q_params[p_name]} '
               f'as min/max is {torch.min(p.detach())}/{torch.max(p.detach())}')

        # register new float32 weight
        module.register_parameter(p_name + '_float32', nn.Parameter(p.detach()))

        # remove the original parameter
        delattr(module, p_name)

    def _apply_replace_with_quantized_weights(self, p_name: str, p: nn.Parameter, module: nn.Module):
        """
        Replace the current parameter with its quantized version.

        This is an apply_fn for the method _iterate_modules_recursive.

        Args:
            p_name: parameter name
            p: apply the function to this parameter
            module: the module
        """
        # get quantization parameters for this module
        q_params = module.q_params[p_name.replace('_float32', '')]

        # apply quantization function
        setattr(module, p_name.replace('_float32', ''), self.q_op.apply(p, *q_params))

    def replace_with_quantized_weights(self):
        """
        Substitute all weights with their quantized counterpart.

        PyTorch forward pre-hook.
        This must receive the same inputs of forward, even if they are not used.
        """
        self._iterate_modules_recursive(self, self._apply_replace_with_quantized_weights)

    def quantize(self, q_op, q_bw):
        """
        Insert the quantization operator in the computational graph.

        Args:
            q_op: quantization operator
            q_bw: quantization bit width
        """
        # store quantization parameters
        self.q_op = q_op
        self.q_bw = q_bw

        uprint(f'Quantizing neural network weights '
               f'(q_op = {q_op}, q_bw = {q_bw}):')

        # insert soft weights and remove original weights
        self._iterate_modules_recursive(self, self._insert_soft_weights)

        uprint('')

    def _convert_to_required_types(self, x: dict, required_types: set):
        """
        Convert a variable to the required types.

        Args:
            x: variable to convert
            required_types: types to convert into

        Raises:
            ValueError: wrong parameters.

        Returns:
             the converted variable
        """
        if DataType.TIME_SAMPLES in required_types and DataType.TIME_SAMPLES not in x:
            if DataType.STFT_COMPLEX in x:
                # compute time-domain signal from real/imaginary part
                x[DataType.TIME_SAMPLES] = spectral_to_time_domain(x[DataType.STFT_COMPLEX],
                                                                   DataType.STFT_COMPLEX,
                                                                   self._stft_window,
                                                                   self._stft_fft_size,
                                                                   self._stft_hop_length)
            elif DataType.STFT_MAGNITUDEPHASE in x:
                # compute time-domain signal from magnitude/phase
                x[DataType.TIME_SAMPLES] = spectral_to_time_domain(x[DataType.STFT_MAGNITUDEPHASE],
                                                                   DataType.STFT_MAGNITUDEPHASE,
                                                                   self._stft_window,
                                                                   self._stft_fft_size,
                                                                   self._stft_hop_length)
            else:
                raise ValueError('Computation of time-domain output not possible'
                                 ' as no suitable STFT representation was found.')

        if DataType.STFT_COMPLEX in required_types and DataType.STFT_COMPLEX not in x:
            if DataType.STFT_MAGNITUDEPHASE in x:
                x[DataType.STFT_COMPLEX] = mp2ri(x[DataType.STFT_COMPLEX])
            elif DataType.TIME_SAMPLES in x:
                x[DataType.STFT_COMPLEX] = time_domain_to_spectral(x[DataType.TIME_SAMPLES],
                                                                   DataType.STFT_COMPLEX,
                                                                   self._stft_window,
                                                                   self._stft_fft_size,
                                                                   self._stft_hop_length)
            else:
                raise ValueError('Computation of STFT real/imaginary output not possible'
                                 ' as no suitable time-domain or STFT representation was found.')

        if DataType.STFT_MAGNITUDEPHASE in required_types and DataType.STFT_MAGNITUDEPHASE not in x:
            if DataType.STFT_COMPLEX in x:
                x[DataType.STFT_MAGNITUDEPHASE] = ri2mp(x[DataType.STFT_COMPLEX])
            elif DataType.TIME_SAMPLES in x:
                x[DataType.STFT_MAGNITUDEPHASE] = time_domain_to_spectral(x[DataType.TIME_SAMPLES],
                                                                          DataType.STFT_MAGNITUDEPHASE,
                                                                          self._stft_window,
                                                                          self._stft_fft_size,
                                                                          self._stft_hop_length)
            else:
                raise ValueError('Computation of STFT magnitude/phase output not possible'
                                 ' as no suitable time-domain or STFT representation was found.')

        if DataType.STFT_MAGNITUDE in required_types and DataType.STFT_MAGNITUDE not in x:
            if DataType.STFT_MAGNITUDEPHASE in x:
                x[DataType.STFT_MAGNITUDE] = x[DataType.STFT_MAGNITUDEPHASE][..., 0]
            elif DataType.STFT_COMPLEX in x:
                x[DataType.STFT_MAGNITUDE] = ri2m(x[DataType.STFT_COMPLEX])
            elif DataType.TIME_SAMPLES in x:
                x[DataType.STFT_MAGNITUDE] = time_domain_to_spectral(x[DataType.TIME_SAMPLES],
                                                                     DataType.STFT_MAGNITUDE,
                                                                     self._stft_window,
                                                                     self._stft_fft_size,
                                                                     self._stft_hop_length)
            else:
                raise ValueError('Computation of STFT magnitude output not possible'
                                 ' as no suitable time-domain or STFT representation was found.')

        return x

    def preprocess(self, x, losses: dict, output_data_type=None):
        """
        Preprocess the variable before the forward pass.

        Args:
            x: input variable
            losses: dict of loss functions to compute

        Returns:
            the preprocessed input variable, its original length, the required output types, the input type
        """
        # We denote S = n_targets

        # flatten parameters for GRU/LSTM cells - needed for multi-GPU training
        for m in self.net.children():
            if isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                m.flatten_parameters()

        # get all data representations that we need for `losses`
        required_output_types = set()
        if losses:
            for loss in losses.values():
                if type(loss.data_type) is set:
                    required_output_types.update(loss.data_type)
                else:
                    required_output_types.add(loss.data_type)
        elif output_data_type:
            required_output_types.add(output_data_type) 

        # make time axis the last one
        x = x.permute((0, 1, 3, 2)).contiguous()  # (1+S)xBxTxC -> (1+S)xBxCxT

        _input_type = None
        if self.net.input_type in [DataType.STFT_MAGNITUDE,
                                   DataType.STFT_MAGNITUDEPHASE,
                                   DataType.STFT_COMPLEX]:
            # check whether we additionally need to compute the mixture phase
            if (self.net.output_type == DataType.STFT_MAGNITUDE and
                (DataType.STFT_COMPLEX in required_output_types or
                 DataType.STFT_MAGNITUDEPHASE in required_output_types or
                 DataType.TIME_SAMPLES in required_output_types)):
                # we need the mixture phase for the iSTFT as the network only provides the magnitude
                # therefore compute it here
                _input_type = DataType.STFT_MAGNITUDEPHASE
                if self.unfolding_params is not None:
                    self.unfolding_params['input_type'] = DataType.STFT_MAGNITUDEPHASE
            else:
                _input_type = self.net.input_type

            # convert to STFT domain
            x = time_domain_to_spectral(x, _input_type,
                                        self._stft_window, self._stft_fft_size,
                                        self._stft_hop_length)  # (1+S)xBxCxFxN(x2)

        # get length of the mixture in its datatype (`_input_type` edge case,
        # i.e. when `x[0]` datatype is not self.net.input_type, does not impact computation)
        original_length = get_length(x[0], self.net.input_type)

        # for validation, transform tracks so that their length match the one
        # of the training samples
        if self.batched_valid is True and not self.training:
            # unfold works only for DataType compliant data
            x = torch.stack([unfold(_x, **self.unfolding_params) for _x in x.unbind(0)], dim=0)

        return x, original_length, required_output_types, _input_type

    def forward(self, x, losses: dict):
        """
        Forward pass.

        Args:
            x: input batch
            losses: dictionary of losses to compute

        Raises:
            ValueError: unknown data type.

        Returns:
            dictionary of losses
        """
        
        with amp.autocast(enabled=self.use_amp):
            # x : (1 + n_targets x batch_size x time_length x n_channels)
#             print('data', x.shape)
            n_stems = self.net.n_stems
            # do preprocessing
            x, original_length, required_output_types, _input_type = self.preprocess(x, losses)
            # perform forward pass through separation network
            out, tar = {}, {}
            if self.net.input_type == DataType.TIME_SAMPLES:
                # time-domain network - we can directly forward through the network
                inp = torch.unbind(x[-n_stems:])
                inp = torch.hstack(inp)
                out[DataType.TIME_SAMPLES] = self.batched_forward(inp, self.net)
                tar[DataType.TIME_SAMPLES] = x[:-n_stems]
            elif self.net.input_type == DataType.STFT_MAGNITUDE:
                if _input_type == self.net.input_type:
                    # we only computed the magnitude as no loss is requiring it
                    out[DataType.STFT_MAGNITUDE] = self.batched_forward(x[0], self.net)
                    tar[DataType.STFT_MAGNITUDE] = x[1:]
                else:
                    # extract magnitude, forward it through network and combine with mixture phase
                    out[DataType.STFT_MAGNITUDEPHASE] =\
                        torch.stack([self.batched_forward(x[0, ..., 0], self.net),
                                     x[0, ..., 1].expand(x[1:, ..., 1].shape)], dim=-1)
                    tar[DataType.STFT_MAGNITUDEPHASE] = x[1:]
            elif self.net.input_type == DataType.STFT_COMPLEX:
                out[DataType.STFT_COMPLEX] = self.batched_forward(x[0], self.net)
                tar[DataType.STFT_COMPLEX] = x[1:]
            elif self.net.input_type == DataType.STFT_MAGNITUDEPHASE:
                out[DataType.STFT_MAGNITUDEPHASE] = self.batched_forward(x[0], self.net)
                tar[DataType.STFT_MAGNITUDEPHASE] = x[1:]
            else:
                raise ValueError('Unknown data type.')

            # for validation, reconstruct estimated tracks from their batched version
            if self.batched_valid is True and not self.training:
                for data_type in out:  # out/tar sole key is not necessarily self.net.input_type
                    out[data_type] = reconstruct_from_unfold(out[data_type],
                                                             original_length=original_length,
                                                             **self.unfolding_params)
                    tar[data_type] = reconstruct_from_unfold(tar[data_type],
                                                             original_length=original_length,
                                                             **self.unfolding_params)

            # compute required representations for the losses (for `out` and `tar`)
            out = self._convert_to_required_types(out, required_output_types)
            tar = self._convert_to_required_types(tar, required_output_types)
            
            # compute losses; the `unsqueeze` is needed for gather
            losses = {key: loss(out, tar).unsqueeze(0) for key, loss in losses.items()}
            return losses
        
    def evaluate(self, x, losses):
        """
        Forward pass.

        Args:
            x: input batch
            losses: dictionary of losses to compute

        Raises:
            ValueError: unknown data type.

        Returns:
            dictionary of losses
        """
        
        with amp.autocast(enabled=self.use_amp):
            # x : (1 + n_targets x batch_size x time_length x n_channels)
            n_stems = self.net.n_stems
            # do preprocessing
            x, original_length, required_output_types, _input_type = self.preprocess(x, losses)
            # perform forward pass through separation network
            out, tar = {}, {}
            if self.net.input_type == DataType.TIME_SAMPLES:
                # time-domain network - we can directly forward through the network
                inp_ = torch.unbind(x[-n_stems:])
                inp = torch.hstack(inp_)
                out[DataType.TIME_SAMPLES] = self.batched_forward(inp, self.net)
                tar[DataType.TIME_SAMPLES] = x[:-n_stems]
            else:
                raise ValueError('Unknown data type.')

            # for validation, reconstruct estimated tracks from their batched version
            if self.batched_valid is True and not self.training:
                for data_type in out:  # out/tar sole key is not necessarily self.net.input_type
                    out[data_type] = reconstruct_from_unfold(out[data_type],
                                                             original_length=original_length,
                                                             **self.unfolding_params)
                    tar[data_type] = reconstruct_from_unfold(tar[data_type],
                                                             original_length=original_length,
                                                             **self.unfolding_params)

            # compute required representations for the losses (for `out` and `tar`)
            
            out = self._convert_to_required_types(out, required_output_types)
            tar = self._convert_to_required_types(tar, required_output_types)
    
#             # compute losses; the `unsqueeze` is needed for gather
            losses = {key: loss(out, tar).unsqueeze(0) for key, loss in losses.items()}

            return tar, out, losses
    
    def inference(self, x):
        """
        Forward pass.

        Args:
            x: input batch

        Raises:
            ValueError: unknown data type.

        Returns:
            output audio
        """
        
        with amp.autocast(enabled=self.use_amp):
            # x : (1 + n_targets x batch_size x time_length x n_channels)
            n_stems = self.net.n_stems
            # do preprocessing
            x, original_length, required_output_types, _input_type = self.preprocess(x,
                                                                                     losses={}, 
                                                                                     output_data_type=DataType.TIME_SAMPLES)
            # perform forward pass through separation network
            out = {}
            if self.net.input_type == DataType.TIME_SAMPLES:
                # time-domain network - we can directly forward through the network
                inp_ = torch.unbind(x[-n_stems:])
                inp = torch.hstack(inp_)
                out[DataType.TIME_SAMPLES] = self.batched_forward(inp, self.net)
            else:
                raise ValueError('Unknown data type.')

            # for validation, reconstruct estimated tracks from their batched version
            if self.batched_valid is True and not self.training:
                for data_type in out:  # out/tar sole key is not necessarily self.net.input_type
                    out[data_type] = reconstruct_from_unfold(out[data_type],
                                                             original_length=original_length,
                                                             **self.unfolding_params)
                    
            # compute required representations for the losses (for `out` and `tar`)
#             out = self._convert_to_required_types(out, required_output_types)

            return out

