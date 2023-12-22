"""
Define the structure of the proposed network "FxNorm Automix"

OURS = CAFX + ConvTasNet + BLSTM

CAFX -> https://www.mdpi.com/2076-3417/10/2/638
ConvTasNet -> https://arxiv.org/abs/1809.07454

AI Music Technology Group, Sony Group Corporation
AI Speech and Sound Group, Sony Europe
"""

import torch
import torch.nn as nn

import numpy as np

from automix.common_datatypes import DataType
from automix.common_supernet import SeparationNet
from automix.common_miscellaneous import pad_to_shape

def compute_receptive_field(KERNEL_SIZE_ENCODER,
                                KERNEL_SIZE_TB,
                                N_TB_PER_REPEAT, 
                                N_REPEATS, MAX_POOLING):
        
        # Length of the encoder's filters, corresponding stride is half of it (L)
        L = KERNEL_SIZE_ENCODER
        stride_enc = 1
        stride_dec = 1
        # Kernel size in convolutional blocks (P)
        P = KERNEL_SIZE_TB
        # Number of temporal blocks in each repeat (X)
        X = N_TB_PER_REPEAT
        # Number of repeat (R)
        R = N_REPEATS
        
        M = MAX_POOLING

#         X_2 = X//2
#         # Number of repeat (R_2)
#         R_2 = R//2
#         stride_dec_2 = stride_dec

#         guard = ((R * (P - 1) // 2 * 2**X - 1) * L//2 + L)

        # RF due to the dilations
        d = []
        for i in range(X):
            d.append(np.power(2,i))
        RF = 1 + R*(P-1)*np.sum(d)
        
        guard = RF *M
#         RF due to the dilations+encoder
        guard = stride_enc * guard + (L - stride_enc)
#         RF due to the decoder
        guard = guard + (stride_enc)*(L - 1)

#         d = []
#         for i in range(X_2):
#             d.append(np.power(2,i))
#         sum_ = (1 + R_2*(P-1)*np.sum(d))
#         # RF due to the new dilations
#         RF = RF + (stride_enc)*(stride_dec)*(sum_ - 1)
#         # RF due to the new decoder
#         RF = RF + (stride_enc)*(stride_dec)*(L - 1)

        return int(RF), guard

class SqueezeExcitation(nn.Module):
    """
    Implementation of Squeeze-and-Excitation (SE) block described in [1].

    References:
        [1] J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu,
        “Squeeze-and-Excitation Networks,” arXiv:1709.01507 [cs], May 2019, Available: http://arxiv.org/abs/1709.01507
    """

    def __init__(self,
                 n_channels,
                 amplifying_ratio) -> None:
        """
        Initialize the layer.

        Args:
            n_channels: Number of input channels.
            amplifying_ratio: Ratio by how much the num_channels should be expanded.
        """
        super(SqueezeExcitation, self).__init__()

        self.n_channels = n_channels
        self.amplifying_ratio = amplifying_ratio
        n_channels_expanded = self.amplifying_ratio * self.n_channels

        self.net = nn.Sequential(nn.Linear(self.n_channels, n_channels_expanded, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(n_channels_expanded, self.n_channels, bias=True),
                                 nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input batch, signal with shape [batch_size x n_channels x n_features]

        Returns:
            x (torch.Tensor): Applied channel excitation.
        """
        x_in = x
        # Absolute value:
        x = torch.abs(x)
        # Global avergage pooling:
        x = x.mean(dim=-1)
        # Channel excitation:
        x = self.net(x).unsqueeze(-1)
        x = x_in * x
        return x


class TemporalBlock(nn.Module):
    """
    Define a temporal block.

    The temporal block preserves the length of the signal received as input and is defined as in figure 1.C of [1].

    Input:
        `input`: the output residual of a preceding TemporalBlock

    Outputs:
        `input + residual`: fresidual features to be passed to the next TemporalBlock

        `skip`            : Skip connection to be accumulated for all temporal
                            blocks, to be used in the later steps of the model
    """

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 sc_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 end_block):
        """
        Initialize the network.

        Args:
            in_channels: number of input channels
            hidden_channels: number of hidden channels
            sc_channels: number of skip channels
            kernel_size: kernel size
            stride: stride
            padding: padding
            dilation: dilation
            end_block: end block is used to have only skip connection at the end
        """
        super(TemporalBlock, self).__init__()

        self.kernel_size = kernel_size
        self.end_block = end_block
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.nonlinearity1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channels, eps=1e-08)
        self.depthwise_conv = nn.Conv1d(hidden_channels,
                                        hidden_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=hidden_channels
                                        )
        self.nonlinearity2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channels, eps=1e-08)
        self.skip_out = nn.Conv1d(hidden_channels, sc_channels, kernel_size=1)
        # always defined to please torch.jit.script, this is suboptimal
        self.res_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, inp):
        """
        Forward pass.

        Args:
            inp: input batch, mixture STFT amplitude with shape [batch_size x n_channels x n_freqbins x n_frames]

        Returns:
            temporal block output
        """
        output = self.norm1(self.nonlinearity1(self.conv1x1(inp)))
        output = self.norm2(self.nonlinearity2(self.depthwise_conv(output)))
        skip = self.skip_out(output)

        if self.end_block is False:
            residual = self.res_out(output)
            return inp + residual, skip
        else:
            # return 0 to make torch.jit.script happy
            return torch.tensor(0), skip


class Net(SeparationNet):
    """
    ConvTasNet vanilla network.

    Input: (batch_size, n_channels, n_samples)
    Output: (n_targets, batch_size, n_channels, n_samples)

    Parameters
    ----------
    N_FEATURES_ENCODER : int
        Number of features after the encoder
    KERNEL_SIZE_ENCODER : int
        Length of the encoder's filters
    N_FEATURES_SEPARATION_MODULE : int
        Number of features in the separation module, after bottleneck layer
    N_FEATURES_OUT : int
        Number of features at the output of the skip connection path
    N_FEATURES_TB : int
        Number of features in the temporal blocks
    KERNEL_SIZE_TB : int
        Kernel size in the temporal blocks
    N_TB_PER_REPEAT : int
        Number of temporal blocks in each repeat
    N_REPEATS : int
        Number of repeat
    """

    input_type = DataType.TIME_SAMPLES
    output_type = DataType.TIME_SAMPLES
    joint_model = True

    def __init__(self,
                 input_offset, input_scale,
                 output_offset, output_scale,
                 n_channels, n_targets, n_stems,
                 N_FEATURES_ENCODER,
                 KERNEL_SIZE_ENCODER,
                 N_FEATURES_SEPARATION_MODULE,
                 N_FEATURES_OUT,
                 N_FEATURES_TB,
                 KERNEL_SIZE_TB,
                 N_TB_PER_REPEAT,
                 N_REPEATS,
                 PRETRAIN, 
                 MAX_POOLING,
                 SE_AMP_RATIO,
                 **kwargs):
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
        self.n_features_encoder = N_FEATURES_ENCODER
        self.kernel_size_encoder = KERNEL_SIZE_ENCODER
        self.n_tb_per_repeat = N_TB_PER_REPEAT
        self.n_repeats = N_REPEATS

        self.n_channels = n_channels
        self.n_stems = n_stems
        
        self.n_filters = N_FEATURES_ENCODER
        self.kernel_size = KERNEL_SIZE_ENCODER
        self.pretrain = PRETRAIN
        self.maxpoolsize = MAX_POOLING
        self.se_amp_ratio = SE_AMP_RATIO

        # Separation part
        self.layerNorm = nn.GroupNorm(1, N_FEATURES_ENCODER, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(N_FEATURES_ENCODER, N_FEATURES_SEPARATION_MODULE, 1)

        self.repeats = nn.ModuleList()

        for r in range(N_REPEATS):
            repeat = nn.ModuleList()
            for x in range(N_TB_PER_REPEAT):
                dilation = 2**x
                padding = (KERNEL_SIZE_TB - 1) * dilation // 2
                repeat.append(TemporalBlock(
                        in_channels=N_FEATURES_SEPARATION_MODULE,
                        hidden_channels=N_FEATURES_TB,
                        sc_channels=N_FEATURES_OUT,
                        kernel_size=KERNEL_SIZE_TB,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        end_block=(r == N_REPEATS - 1 and x == N_TB_PER_REPEAT - 1)
                                            )
                              )

            self.repeats.append(repeat)

        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(N_FEATURES_OUT,
                                              n_targets * N_FEATURES_ENCODER,
                                              kernel_size=1),
                                    nn.Softplus()
                                    )
   


        
        # ----- Adaptive front-end ----- #

        self.conv_1 = nn.Conv1d(in_channels=n_channels*n_stems,
                                out_channels=N_FEATURES_ENCODER,
#                                 groups=n_stems,
                                kernel_size=KERNEL_SIZE_ENCODER, bias=False,
                                stride=1,
                                padding='same')

        self.pad = [pad * -1 for pad in self.conv_1._reversed_padding_repeated_twice]

        self.conv_2 = nn.Sequential(nn.Conv1d(in_channels=N_FEATURES_ENCODER,
                                              out_channels=N_FEATURES_ENCODER,
                                              groups=N_FEATURES_ENCODER,
                                              kernel_size=KERNEL_SIZE_ENCODER*2, bias=False,
                                              stride=1,
                                              padding='same'),
                                    nn.Softplus())

        self.maxpool1d = nn.MaxPool1d(kernel_size=self.maxpoolsize, return_indices=True)
        
        # ----- Latent-space Bi-LSTM (not used when pre-training) ----- #
        
        self.blstm = nn.LSTM(N_FEATURES_ENCODER,
                             N_FEATURES_ENCODER//2,
                             num_layers=3,
                             batch_first=True,
                             dropout=0.1, 
                             bidirectional=True)

        
#         self.blstm = nn.ModuleList([nn.LSTM(input_size=self.n_filters*self.n_context_frames_total,
#                                             hidden_size=self.n_filters*2,
#                                             bias=True,
#                                             batch_first=True,
#                                             bidirectional=True),
#                                     nn.LSTM(input_size=self.n_filters*4,
#                                             hidden_size=self.n_filters,
#                                             bias=True,
#                                             batch_first=True,
#                                             bidirectional=True),
#                                     nn.LSTM(input_size=self.n_filters*2,
#                                             hidden_size=self.n_filters//2,
#                                             bias=True,
#                                             batch_first=True,
#                                             bidirectional=True)])

        
        # ----- Synthesis back-end ----- #

        self.maxunpool1d = nn.MaxUnpool1d(self.maxpoolsize)
        
        self.se_block = SqueezeExcitation(n_channels=N_FEATURES_ENCODER, amplifying_ratio=self.se_amp_ratio) 
        
        self.tanh = nn.Tanh()

        # Specify that gradient should not be computed for those layers during pretraining:
        if self.pretrain:
            for param in self.bottleneck_conv1x1.parameters():
                param.requires_grad = False
            for param in self.layerNorm.parameters():
                param.requires_grad = False
            for param in self.repeats.parameters():
                param.requires_grad = False
            for param in self.output.parameters():
                param.requires_grad = False
            for param in self.se_block.parameters():
                param.requires_grad = False
                
        self.reset_parameters()
        
    def reset_parameters(self, heuristic='init_chrono'):
        """
        Initialize forget gate according to some heuristic.

            `init_1`:      initialize forget gate to 1.0
            `init_chrono`: https://arxiv.org/abs/1804.11188

        Args:
            heuristic: which heuristic to use for initialization, see above for options
        """
        if heuristic is not None:
            lstm_layers = [self.blstm]
            for l in lstm_layers:
                for names in l._all_weights:
                    for name in filter(lambda n: 'bias' in n,  names):
                        bias = getattr(l, name)
                        n = bias.size(0) // 4
                        if heuristic == 'init_1':
                            bias.data[n:2*n].fill_(1.)
                        elif heuristic == 'init_chrono':
                            Tmin = 1
                            Tmax = 32
                            # initialize everything to zero
                            bias.data.fill_(0.)
                            # forget gate biases = log(uniform(1, Tmax-1))
                            bias.data[n:2*n] = \
                                torch.log(nn.init.uniform_(bias.data[0:n], Tmin, Tmax-1))
                            # input gate biases = -(forget gate biases)
                            bias.data[0:n] = -bias.data[n:2*n]

                            
    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input batch, mixture STFT amplitude with shape [batch_size x n_channels x n_freqbins x n_frames]

        Returns:
            source STFT amplitude with same dimensions as input
        """
        batch_size, _, n_samples = x.shape
        
#         print('input', x.size()) # batch_size, channels * stems, samples
       


        # ----- Adaptive front-end ----- #

        x = self.conv_1(x)

        p = x
        x = torch.abs(x)
        x = self.conv_2(x)

        x, pool_indices = self.maxpool1d(x)
        
        if not self.pretrain:

            # --- SEPARATION ---
            # output [batch_size, N_FEATURES_ENCODER, hidden_size]
            masking = self.layerNorm(x)

            # output [batch_size, N_FEATURES_SEPARATION_MODULE, hidden_size]
            masking = self.bottleneck_conv1x1(masking)

            skip_connection = 0.
            for repeat in self.repeats:
                for temporalBlock in repeat:
                    # `masking` corresponds to the residual path that serves as
                    # an output to the next temporal block, while skip connections
                    # are accumulated for all blocks, and are used in the later steps

                    # masking [batch_size, N_FEATURES_OUT, hidden_size]
                    # skip [batch_size, N_FEATURES_SEPARATION_MODULE, hidden_size]
                    masking, skip = temporalBlock(masking)
                    skip_connection = skip_connection + skip

            # output [batch_size, n_targets*n_features_encoder, hidden_size]
            masking = self.output(skip_connection)
            masking = masking.transpose(1,2)

            masking = self.blstm(masking)[0]
            masking = nn.Softplus()(masking)
        
            masking = masking.transpose(1,2)

            masking = nn.functional.interpolate(masking, scale_factor=self.maxpoolsize)
            
        else:
            masking = self.maxunpool1d(x, pool_indices)
            
        # output [batch_size, n_targets, n_features_encoder, hidden_size]
        masking = masking.view(batch_size, self.n_targets,
                               self.n_features_encoder, masking.shape[-1])

        # output [batch_size, n_targets, n_features_encoder, hidden_size]

        x = masking * torch.unsqueeze(p, 1)
        

        

#         # --- DECODER ---
#         # output [batch_size * n_targets, n_features_encoder, hidden_size]
        x = x.view(batch_size * self.n_targets, self.n_features_encoder, -1)

        if not self.pretrain:
            x = self.se_block(x)


        # De-convolution (use transposed kernel from conv_1, additionally learn the bias)
        x = nn.functional.pad(x, self.pad)
        x = nn.functional.conv_transpose1d(x, self.conv_1.weight)

        
        x_stems = x.view(batch_size, self.n_stems, self.n_channels, -1)

        # output [n_targets, batch_size, n_channels, n_samples_possibly_diff]
        x_stems = torch.transpose(x_stems, 0, 1)

        # output [n_targets, batch_size, n_channels, n_samples]
        # this padding is required because the n_samples would otherwise be
        # changed due to the encoder
        
        if self.pretrain:
            x_stems = pad_to_shape(x_stems, n_samples)
            return x_stems
        
        else:
            x_stems_ = torch.unbind(x_stems)
            x_mix = torch.stack(x_stems_, dim=0).sum(dim=0, keepdim=True)

    #         x_stems = pad_to_shape(x_stems, n_samples)
            x_mix = pad_to_shape(x_mix, n_samples)

            x_mix = self.tanh(x_mix)

    #         out = torch.cat((x_mix, x_stems), dim=0)


            return x_mix
