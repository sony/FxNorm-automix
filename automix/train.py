"""
Train an automix model.

Usage:
  python train.py CONFIG.py
                 [--folder-suffix FOLDER_SUFFIX]
                 [--description DESCRIPTION]
                 [--results-folder RESULTS_FOLDER]
                 [--weight-initialization PARAMS_NET1.params ...]

`FOLDER_SUFFIX` is appended to the folder name which is stored in `Results`,
i.e., the folder name looks like `{Date}_PT{torch.__version__}_{EXPERIMENT_STRING}`.

`DESCRIPTION` is a textual description of the experiment which you can use to log
what kind of experiment you are launching.

RESULTS_FOLDER the path where the results are saved   

AI Music Technology Group, Sony Group Corporation
AI Speech and Sound Group, Sony Europe
"""
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Process, SimpleQueue, RawArray, Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import os
from pprint import pformat
import random
import socket
import subprocess
from shutil import copyfile
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import sklearn

from automix.common_dataprocessing import create_dataset_mixing, create_minibatch_mixing
from automix.common_datatypes import DataType
from automix.common_miscellaneous import uprint, recursive_getattr, get_process_memory, compute_stft
from automix.common_miscellaneous import td_length_from_fd
from automix.common_supernet import SuperNet

plt.switch_backend('agg')
sys.setrecursionlimit(int(1e6))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Training parser')

parser.add_argument('config_file', help='Configuration file for training.')

parser.add_argument('--folder-suffix', type=str, default=None, required=False,
                    help=('Suffix that is appended to the folder in `./Results`. '
                          'If none is provided then it is derived from the config-file name'))

parser.add_argument('--results-folder', type=str,
                    help='Results folder of the experiment.')

parser.add_argument('--description', type=str, default=None, required=False,
                    help='Description of the experiment.')

parser.add_argument('--weight-initialization', type=str, default=None, required=False, nargs='+',
                    help='Path to `.params` files which should be used as network initializations.')

args = parser.parse_args()


if args.folder_suffix is None:
    args.folder_suffix = os.path.splitext(os.path.basename(args.config_file))[0]

# create a unique folder where to store log file and trained models
results_folder = f'{args.results_folder}/results/{time.strftime("%Y%m%d-h%Hm%Ms%S")}_PT{torch.__version__}_{args.folder_suffix}'
os.mkdir(results_folder)

# store config file in result folder (to reproduce training if necessary)
copyfile(args.config_file, os.path.join(results_folder, 'config.py'))

# use `tee` to duplicate `stdout` and `stderr` to `training.log` inside the results folder
tee = subprocess.Popen(['tee', os.path.join(results_folder, 'training.log')], stdin=subprocess.PIPE)
os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

uprint('Starting training:')
uprint(f'\t        Results folder: {results_folder}')
uprint(f'\t    Configuration file: {args.config_file}')
uprint(f'\tExperiment description: {args.description}')
uprint('')

# Read configuration script
config, Net = {}, None
exec(open(args.config_file).read())

# Specify mapping of sources to devices
# (if the user specified `CUDA_VISIBLE_DEVICES`, then make use of it)
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    ngpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
else:
    hostname = socket.gethostname()
    ngpus = subprocess.run(['nvidia-smi', '-L'],
                           capture_output=True).stdout.count(b'\n')

if config['TENSORBOARD']:
    from torch.utils.tensorboard import SummaryWriter

# May help with computation speed for convolutional networks
if config['CUDNN_BENCHMARK']:
    torch.backends.cudnn.benchmark = True

# compute how many GPUs each source will receive - at least one GPU per source
ngpus_per_src = max(1, ngpus // len(config['TARGETS']))

# compute sequence length in the time-domain (for data generation)
if Net.input_type == DataType.TIME_SAMPLES:
    seq_length_td = config['TRAINING_SEQ_LENGTH']
else:
    seq_length_td = td_length_from_fd(config['TRAINING_SEQ_LENGTH'], fft_size=config['FFT_SIZE'],
                                      fft_hop=config['HOP_LENGTH'])

# dump settings to `settings.log`
with open(os.path.join(results_folder, 'settings.log'), 'w') as f:
    uprint('\nGlobal variables:')
    uprint(pformat(locals(), indent=4, width=120))
    uprint('')
    f.write('\nGlobal variables:\n')
    f.write(pformat(locals(), indent=4, width=120) + '\n')
    f.write('\n')

    uprint('\nEnvironment variables:')
    uprint(pformat(dict(os.environ), indent=4, width=120))
    uprint('')
    f.write('\nEnvironment variables:\n')
    f.write(pformat(dict(os.environ), indent=4, width=120) + '\n')
    f.write('\n')



uprint(f'Maximum validation length is '
       f'{config["MAX_VALIDATION_SEQ_LENGTH_TD"] / np.max(config["ACCEPTED_SAMPLING_RATES"]) / 60.:.2f}m')

# Prepare data (train)
uprint('Create dataset (train) ...')
start_time = time.time()
data_train = []
for path, load_into_memory in config['DATA_DIR_TRAIN']:
    data_train.extend(create_dataset_mixing(path=path,
                                     accepted_sampling_rates=config['ACCEPTED_SAMPLING_RATES'],
                                     sources=config['SOURCES'],
                                     mapped_sources=config['MAPPED_SOURCES'],
                                     n_channels=config['N_CHANNELS'],
                                     load_to_memory=load_into_memory,
                                     debug=config['DEBUG'])[0])
uprint(f'\ttook {time.time() - start_time:.2f}s, '
       f'current memory consumption is {get_process_memory():.2f}GB')
uprint('')

# Prepare data (valid)
uprint('Create dataset (valid) ...')
start_time = time.time()
data_valid = []
for path, load_into_memory in config['DATA_DIR_VALID']:
    data_valid.extend(create_dataset_mixing(path=path,
                                     accepted_sampling_rates=config['ACCEPTED_SAMPLING_RATES'],
                                     sources=config['SOURCES'],
                                     mapped_sources=config['MAPPED_SOURCES'],
                                     n_channels=config['N_CHANNELS'],
                                     load_to_memory=load_into_memory,
                                     debug=config['DEBUG'])[0])
uprint(f'\ttook {time.time() - start_time:.2f}s, '
       f'current memory consumption is {get_process_memory():.2f}GB')
uprint('')

if config['DEBUG'] == False:
    # Compute baseline losses
    if len(data_valid) > 0:
        uprint('Compute baseline losses ...')
        start_time = time.time()
        l1_valid_baseline_mix_time = [dict() for _ in range(len(data_valid))]
        l2_valid_baseline_mix_time = [dict() for _ in range(len(data_valid))]
        l1_valid_baseline_mix_freq = [dict() for _ in range(len(data_valid))]
        l2_valid_baseline_mix_freq = [dict() for _ in range(len(data_valid))]
        valid_sample_length = [None] * len(data_valid)

        def compute_baselines(i):
            """
            Compute baselines for each validation song.

            - Baseline `mix`: estimate = mix is summation of stems

            Args:
                i: song index
            """
            # get minimum size
            min_size = np.iinfo(np.int32).max
            data_key = list(data_valid[i])[0]
            min_size = np.minimum(min_size, data_valid[i][data_key]()[0].shape[0])

            # make sure that audio is not too long
            min_size = np.minimum(config['MAX_VALIDATION_SEQ_LENGTH_TD'], min_size)
            min_size -= min_size % config['HOP_LENGTH']  # mimic STFT truncation behavior

            # save size (for sorting later by length)
            valid_sample_length[i] = min_size

            # convert into frequency domain
            stft, mix_f = {}, 0
            samples, mix_t = {}, 0
            audio_tags = data_key.split("-")
            for k, src in enumerate(audio_tags):
                samples[src] = data_valid[i][data_key]()[k][:min_size]
                stft[src] = compute_stft(samples[src],
                                         hop_length=config['HOP_LENGTH'],
                                         fft_size=config['FFT_SIZE'],
                                         stft_window=config['STFT_WINDOW'])

            # compute magnitudes
            for src in audio_tags:
                stft[src] = np.abs(stft[src])
                if src in config['INPUTS']:
                    mix_f += stft[src]
                    mix_t += samples[src]

            # compute lower baselines

            src = config['OUTPUTS'][0]
            l1_valid_baseline_mix_freq[i][src] = np.mean(np.abs(stft[src] - mix_f))
            l2_valid_baseline_mix_freq[i][src] = np.mean(np.square(stft[src] - mix_f))
            l1_valid_baseline_mix_time[i][src] = np.mean(np.abs(samples[src] - mix_t))
            l2_valid_baseline_mix_time[i][src] = np.mean(np.square(samples[src] - mix_t))

            for src, inp in zip(config['OUTPUTS'][1:],config['INPUTS']):
                l1_valid_baseline_mix_freq[i][src] = np.mean(np.abs(stft[src] - stft[inp]))
                l2_valid_baseline_mix_freq[i][src] = np.mean(np.square(stft[src] - stft[inp]))
                l1_valid_baseline_mix_time[i][src] = np.mean(np.abs(samples[src] - samples[inp]))
                l2_valid_baseline_mix_time[i][src] = np.mean(np.square(samples[src] - samples[inp]))

        with ThreadPool(config['NUM_DATAPROVIDING_PROCESSES']) as tp:
            tp.map(compute_baselines, range(len(data_valid)))
    #         print('yes')

        uprint('\tLower baselines on validation dataset (time-domain, l1):')
        uprint('\t\tMean\tMedian')
        for src in config['OUTPUTS']:
            # print table to terminal
            uprint(f'\t{src}'
                   f'\t{np.mean([l1_valid_baseline_mix_time[_][src] for _ in range(len(data_valid))]):.3f}'
                   f'\t{np.median([l1_valid_baseline_mix_time[_][src] for _ in range(len(data_valid))]):.3f}')
        uprint('')
        uprint('\tLower baselines on validation dataset (freq-domain, l1):')
        uprint('\t\tMean\tMedian')
        for src in config['OUTPUTS']:
            # print table to terminal
            uprint(f'\t{src}'
                   f'\t{np.mean([l1_valid_baseline_mix_freq[_][src] for _ in range(len(data_valid))]):.3f}'
                   f'\t{np.median([l1_valid_baseline_mix_freq[_][src] for _ in range(len(data_valid))]):.3f}')
        uprint('')

        uprint('\tLower baselines on validation dataset (time-domain, l2):')
        uprint('\t\tMean\tMedian')
        for src in config['OUTPUTS']:
            # print table to terminal
            uprint(f'\t{src}'
                   f'\t{np.mean([l2_valid_baseline_mix_time[_][src] for _ in range(len(data_valid))]):.3f}'
                   f'\t{np.median([l2_valid_baseline_mix_time[_][src] for _ in range(len(data_valid))]):.3f}')

            # store result (per song) in TXT file
            with open(os.path.join(results_folder, 'VALID_ERROR_L2_' + src + '.txt'), 'a') as f:
                f.write(' '.join([f'{_+1.:6.3f}' for _ in np.arange(len(data_valid), dtype=np.float32)]))
                f.write('\n')
                f.write(' '.join([f'{l2_valid_baseline_mix_time[_][src]:6.3f}' for _ in range(len(data_valid))]))
                f.write('  # time-domain mix baseline\n')
        uprint('')
        uprint('\tLower baselines on validation dataset (freq-domain, l2):')
        uprint('\t\tMean\tMedian')
        for src in config['OUTPUTS']:
            # print table to terminal
            uprint(f'\t{src}'
                   f'\t{np.mean([l2_valid_baseline_mix_freq[_][src] for _ in range(len(data_valid))]):.3f}'
                   f'\t{np.median([l2_valid_baseline_mix_freq[_][src] for _ in range(len(data_valid))]):.3f}')

            # store result (per song) in TXT file
            with open(os.path.join(results_folder, 'VALID_ERROR_L2_' + src + '.txt'), 'a') as f:
                f.write(' '.join([f'{l2_valid_baseline_mix_freq[_][src]:6.3f}' for _ in range(len(data_valid))]))
                f.write('  # freq-domain baseline\n')
        uprint('')

        # sort validation samples by length (for faster evaluation)
        data_valid = [data_valid[_] for _ in np.argsort(valid_sample_length)[::-1]]
        valid_sample_length = np.sort(valid_sample_length)[::-1]

        uprint(f'\ttook {time.time() - start_time:.2f}s, '
               f'current memory consumption is {get_process_memory():.2f}GB')
        uprint('')


    # Compute mean/scale of mixture (= input)
    def compute_offset_and_scale(i):
        """
        Compute offset and scale for a song.

        Args:
            i: song index

        Raises:
            ValueError: Unsupported data type for offset/scale computation.

        Returns:
            input offset, input scale (mix), number of samples
        """
        # compute mixture
        x = 0.0
        audio_tags = list(data_train[i])[0].split("-")
        for k, tag in enumerate(audio_tags):
            if tag in config['INPUTS']:
                x += data_train[i][list(data_train[i])[0]]()[k]

        if Net.input_type == DataType.TIME_SAMPLES:
            input_offset = x.mean()
            input_scale = x.std()
        elif Net.input_type == DataType.STFT_MAGNITUDE:
            # compute STFT
            stft = compute_stft(x,
                                hop_length=config['HOP_LENGTH'],
                                fft_size=config['FFT_SIZE'],
                                stft_window=config['STFT_WINDOW'])

            # compute input_offset/input_scale for mixture (magnitude)
            mix_abs = np.abs(stft)
            input_offset = mix_abs.mean(axis=(0, 1))
            input_scale = mix_abs.std(axis=(0, 1))
        elif Net.input_type == DataType.STFT_COMPLEX:
            # compute STFT
            stft = compute_stft(x,
                                hop_length=config['HOP_LENGTH'],
                                fft_size=config['FFT_SIZE'],
                                stft_window=config['STFT_WINDOW'])

            # compute input_offset/input_scale for mixture
            input_offset = stft.mean(axis=(0, 1))
            input_scale = stft.std(axis=(0, 1))

            # map complex mean to same value for real/imaginary part
            input_offset = 0.5 * (np.real(input_offset) + np.imag(input_offset))
        else:
            raise ValueError('Unsupported data type for offset/scale computation.')
        return input_offset, input_scale, x.shape[0]

    if config['CALCULATE_STATISTICS'] == True:
        uprint('Compute mean/scale on training set ...')
        start_time = time.time()

        with Pool(config['NUM_DATAPROVIDING_PROCESSES']) as p:
            _data = p.map(compute_offset_and_scale, range(len(data_train)))

        input_offset, input_scale, num_samples = 0., 0., 0
        for _input_offset, _input_scale, n in _data:
            input_offset += _input_offset
            input_scale += _input_scale
            num_samples += n

        input_offset /= len(data_train)
        input_scale /= len(data_train)
        del _data

    else:
        input_offset = 0
        input_scale =  0

        if config['DEBUG'] == True:
            valid_sample_length = [2*60*44100] * len(data_valid)
        with Pool(config['NUM_DATAPROVIDING_PROCESSES']) as p:
            print('Statistics bypassed')
            
else:
    input_offset = 0
    input_scale =  0
    valid_sample_length = [2*60*44100] * len(data_valid)
    with Pool(config['NUM_DATAPROVIDING_PROCESSES']) as p:
        print('Statistics bypassed')
        

# make sure that we do not input_scale by more than `1e-4 * max_value`
input_scale = np.maximum(input_scale, 1e-4 * np.max(input_scale))


uprint(f'\ttook {time.time() - start_time:.2f}s, '
       f'current memory consumption is {get_process_memory():.2f}GB')
uprint('')


# Create queues and processes that fill them with data
# We use one `RawArray` to share `input` and `target`
uprint('Create shared memory variables ...')
start_time = time.time()

queues_empty, queues_filled, raw_arrays = {}, {}, {}
for src in config['TARGETS']:
    queues_empty[src] = SimpleQueue()
    queues_filled[src] = SimpleQueue()

    raw_arrays[src] = [None] * config['NUM_DATAPROVIDING_PROCESSES']
    n_targets = len(src)  # number of targets P for this model
    n_inputs = len(config['INPUTS'])
    n_outputs = len(config['OUTPUTS'])
    for q in range(config['NUM_DATAPROVIDING_PROCESSES']):
        raw_arrays[src][q] = RawArray('f', (n_inputs + n_outputs) *
                                      seq_length_td * config['BATCH_SIZE'] * config['N_CHANNELS'])
        queues_empty[src].put(q)

uprint(f'\ttook {time.time() - start_time:.2f}s, '
       f'current memory consumption is {get_process_memory():.2f}GB')
uprint('')

# For stratified sampling, also create a queue which contains the indices
# for each source - this will ensure that we see for each source
# all songs before sampling a song again
queue_idx_songs = SimpleQueue()
queue_idx_songs_sentinel = None

queue_idx_songs.put(queue_idx_songs_sentinel)


def is_sentinel(o):
    """
    Check if object is a sentinel.

    As we use multiprocessing, we can not simply use `o is sentinel`.

    Args:
        o: object

    Returns:
        whether the object is a sentinel
    """
    return isinstance(o, type(None))


def fill_queues(random_seed1, random_seed2):
    """
    Fill the RawArrays in `raw_arrays` with training data from `create_minibatch`.

    Data provider function: this is spawned `config['NUM_DATAPROVIDING_PROCESSES']` times.

    Args:
        random_seed1: first random seed
        random_seed2: second random seed
    """
    # make sure that each process has its own random seed
    # please note that seeding `random` is actually not required as it seeds manually
    # after a fork using `_os.register_at_fork(after_in_child=_inst.seed)` in
    # https://github.com/python/cpython/blame/3.9/Lib/random.py but we do it here for consistency.
    # Furthermore, we are not using the same random seed for `random` as the underlying algorithm
    # is the same (https://github.com/PyTorchLightning/pytorch-lightning/pull/6960#issuecomment-818393659)
    np.random.seed(random_seed1)
    random.seed(random_seed2)

    while True:
        # get index of songs that we should consider
        idx_songs = queue_idx_songs.get()

        # if last element, then create new minibatches
        if is_sentinel(idx_songs):
            # randomly permute the indices for each source
            all_idx_songs = {}
            new_idx = np.concatenate([np.random.permutation(len(data_train)) for _ in range(config['BATCH_SIZE'])])
            for src in config['OUTPUTS']:
                all_idx_songs[src] = new_idx

            # create `idx_songs` for one minibatch
            for i in range(len(all_idx_songs[config['OUTPUTS'][0]]) // config['BATCH_SIZE']):
                _idx_songs = {}
                for src in config['OUTPUTS']:
                    _idx_songs[src] = all_idx_songs[src][i*config['BATCH_SIZE']:(i+1)*config['BATCH_SIZE']]
                queue_idx_songs.put(_idx_songs)

            # finally, put in sentinel
            queue_idx_songs.put(queue_idx_songs_sentinel)

            # skip this iteration and try to get again a new `idx_songs`
            continue

        # get data for this minibatch from `generator`
        np_inp, np_tar = create_minibatch_mixing(data=data_train,
                                          sources=config['SOURCES'],
                                          inputs=config['INPUTS'],
                                          outputs=config['OUTPUTS'],
                                          present_prob=config['PRESENT_PROBABILITY'],
                                          overlap_prob=config['OVERLAP_PROBABILITY'],
                                          augmenter=config['AUGMENTER_CHAIN'],
                                          augmenter_padding=config['AUGMENTER_PADDING'],
                                          augmenter_sources=config['AUGMENTER_SOURCES'],
                                          batch_size=config['BATCH_SIZE'],
                                          n_samples=seq_length_td,
                                          n_channels=config['N_CHANNELS'],
                                          idx_songs=idx_songs)
        for src in config['TARGETS']:
            # get index of empty slot in `raw_arrays`
            idx = queues_empty[src].get()

        #     # cast slot into numpy array
            np_array = np.ctypeslib.as_array(raw_arrays[src][idx])
            # `-1` stands for the mixture plus P targets (possibly P > 1)
            np_array.shape = (-1, config['BATCH_SIZE'], seq_length_td, config['N_CHANNELS'])

#             np.copyto(np_array[0], np_tar)

            for i, source_name in enumerate(config['OUTPUTS']):
                np.copyto(np_array[i], np_tar[source_name])

            for i, source_name in enumerate(config['INPUTS']):
                j = len(config['OUTPUTS'])+i
                np.copyto(np_array[j], np_inp[source_name])
                
            # shuffle order!
            if config['SHUFFLE_STEMS']:
                if 'mixture' in config['OUTPUTS']:
                    idx = 1
                else:
                    idx = 0
                n_stems=len(config['INPUTS'])
                inp_ = np_array[-n_stems:]
                channels = np.arange(inp_.shape[0])
                np.random.shuffle(channels)
        
                inp_ = inp_[channels, ...]
                np.copyto(np_array[-n_stems:], inp_)
                
                if len(config['OUTPUTS']) > 1:
                    tar_ = np_array[idx:-n_stems]
                    tar_ = tar_[channels, ...]
                    np.copyto(np_array[idx:-n_stems], tar_)
                
                # shuffle channels!
            if config['SHUFFLE_CHANNELS']:
                channels = np.arange(np_array.shape[-1])
                np.random.shuffle(channels)
                np_array = np_array[..., channels]
                
            # tell training process for this source that this slot in
            # `raw_arrays[src]` is filled
            queues_filled[src].put(idx)


def train_network(my_src):
    """
    Train the neural network.

    This is spawned `len(config['SOURCES'])` times.

    Raises:
        RuntimeError: initialization param not found

    Args:
        my_src: the source assigned to this network
    """
    my_src_idx = config['TARGETS'].index(my_src)
    pretty_my_src = '-'.join([i for i in my_src])  # nicer formatting for file naming

    # get the ids of the GPUs for this training process
    device_ids = [(_ + my_src_idx * ngpus_per_src) % ngpus for _ in range(0, ngpus_per_src)]


    # this avoids that all processes allocate something on GPU 0
    torch.cuda.set_device(device_ids[0])

    # Add Tensorboard
    if config['TENSORBOARD']:
        # tensorboard can not write several summaries to the same directory
        tb_dir = os.path.join(results_folder, 'tb', pretty_my_src)
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        writer_loss = SummaryWriter(log_dir=tb_dir)

    # create `torch.tensors` from RawArrays for this source and pin memory
    data_tensors = [None] * config['NUM_DATAPROVIDING_PROCESSES']
    for q in range(config['NUM_DATAPROVIDING_PROCESSES']):
        # cast slot into numpy array
        np_array = np.ctypeslib.as_array(raw_arrays[my_src][q])
        # `-1` stands for the mixture plus P targets (possibly P > 1)
        np_array.shape = (-1, config['BATCH_SIZE'], seq_length_td, config['N_CHANNELS'])

        # create torch.tensor from numpy array
        data_tensors[q] = torch.from_numpy(np_array)

        # pin their memory
        data_tensors[q].pin_memory()

    # build separation network
    net = Net(input_offset=input_offset, input_scale=input_scale,
              output_offset=np.ones(config['N_BINS'], dtype=np.float32),
              output_scale=np.ones(config['N_BINS'], dtype=np.float32),
              n_channels=config['N_CHANNELS'],
              n_targets=len(my_src),
              n_stems=len(config['INPUTS']),
              **config)

    if config['INIT_NETWORK'] is not None:
        net.initialize_network(heuristic=config['INIT_NETWORK'])

    # save current network structure (training & evaluation), for each network
    torch.save(net, results_folder + '/net_' + pretty_my_src + '.dump')

    np.savez(results_folder + '/training_params.npz',
             N_CHANNELS=config['N_CHANNELS'],
             N_STEMS=len(config['INPUTS']),
             N_BINS=config['N_BINS'],
             FFT_SIZE=config['FFT_SIZE'],
             HOP_LENGTH=config['HOP_LENGTH'],
             STFT_WINDOW=config['STFT_WINDOW'],
             ACCEPTED_SAMPLING_RATES=config['ACCEPTED_SAMPLING_RATES'],
             N_BINS_KEEP=config['N_BINS_KEEP'],
             GUARD_LEFT=config['GUARD_LEFT'],
             GUARD_RIGHT=config['GUARD_RIGHT'],
             TRAINING_SEQ_LENGTH=config['TRAINING_SEQ_LENGTH'],
             TRAINING_BATCH_SIZE=config['BATCH_SIZE'] // ngpus_per_src
             )
    try:
        torch.jit.save(torch.jit.script(net), results_folder + '/net_' + pretty_my_src + '.jit.dump')
        uprint('Successfully saved TorchScript model.')
    except:  # noqa E722
        uprint('WARNING: Could not save TorchScript model.')

    # plot network structure and overall number of parameters
    if my_src == config['TARGETS'][-1]:
        uprint(f'\nNetwork parameters for {config["NET_TYPE"]}:')
        num_overall_parameters = 0
        for name, param in net.named_parameters():
            uprint(f'\t{name:30s}\t{str(list(param.shape)):25s}\t{np.prod(param.shape)}')
            num_overall_parameters += np.prod(param.shape)
        uprint(f'\n\tIn total this network has {num_overall_parameters} parameters.\n')

    # set network parameters (if set by user)
    if args.weight_initialization is not None:
        if os.path.isfile(args.weight_initialization[my_src_idx]):
            uprint(f'Loading weights for {pretty_my_src} from {args.weight_initialization[my_src_idx]}')
            if config['PRETRAIN_FRONT_END']:
                pretrained_dict = torch.load(args.weight_initialization[my_src_idx], map_location='cpu')
                model_dict = net.state_dict()
                # # # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.split('.')[0] in ['conv_1', 'conv_2']}
                # # # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict) 
                # # # 3. load the new state dict
                net.load_state_dict(model_dict)
            else:
                net.load_state_dict(torch.load(args.weight_initialization[my_src_idx],
                                               map_location='cpu'))
        else:
            uprint(f'Could not find weight initialization file {args.weight_initialization[my_src_idx]}')
            raise RuntimeError(f'Could not find weight initialization file {args.weight_initialization[my_src_idx]}')

    unfolding_params = None
    if config['BATCHED_VALID']:
        unfolding_params = {'window_size': config['TRAINING_SEQ_LENGTH'],
                            'guard_left': config['GUARD_LEFT'],
                            'guard_right': config['GUARD_RIGHT'],
                            'input_type': net.input_type}

    # build super-net (doing also input/output data-conversion if needed)
    # For now, we assume to have always set in the config file `FFT_SIZE` and
    # `HOP_LENGTH`, as they correspond to parameters not only for the some models
    # to run, but ** most importantly ** are also used for the loss computations
    super_net = SuperNet(net,
                         stft_window=torch.from_numpy(config['STFT_WINDOW'].astype(np.float32)),
                         stft_hop_length=config['HOP_LENGTH'],
                         batched_valid=config['BATCHED_VALID'],
                         unfolding_params=unfolding_params,
                         training_length=config['TRAINING_SEQ_LENGTH'],
                         training_batch_size=config['BATCH_SIZE'] // ngpus_per_src,
                         use_amp=config['USE_AMP'])

    # transfer model and losses to the GPU
    super_net.to('cuda')
    for loss in config['TRAIN_LOSSES']:
        loss.to('cuda')
    for loss in config['VALID_LOSSES'].values():
        loss.to('cuda')

    if config['QUANTIZATION_OP'] is not None:
        super_net.quantize(config['QUANTIZATION_OP'], config['QUANTIZATION_BW'])

    grad_scaler = amp.GradScaler(enabled=config['USE_AMP'])

    # define optimizer
    learning_rate = np.concatenate([l*np.ones((n,)) for n, l in config['LEARNING_RATES']])
    if config['L2_REGULARIZATION'] is None:
        optimizer = optim.Adam(net.parameters(),
                               lr=learning_rate[0],
                               amsgrad=config['AMSGRAD'])
    else:
        optimizer = optim.Adam(net.parameters(),
                               lr=learning_rate[0],
                               weight_decay=config['L2_REGULARIZATION'],
                               amsgrad=config['AMSGRAD'])

    # initialize arrays to store validation errors
    valid_loss = {_: np.zeros((len(data_valid),)) for _ in config['VALID_LOSSES']}
    best_valid_mean, best_valid_median = {}, {}
    for key in config['VALID_LOSSES']:
        best_valid_mean[key], best_valid_median[key] = np.inf, np.inf

    # Prepare the dictionaries with validation losses
    val_losses = [dict() for _ in device_ids]
    for key in config['VALID_LOSSES']:
        for idx, repl in enumerate(nn.parallel.replicate(config['VALID_LOSSES'][key], device_ids)):
            val_losses[idx][key] = repl

    def parallel_perform_valid(data_indices, replicas):
        """
        Perform validation.

        Args:
            data_indices: indices of validation data
            replicas: replicas of the network (for DataParallel)
        """
        # prepare the validation data
        val_data = []  # will contain the validation data, each with different sequence length
        for i in data_indices:
            # create input/target
            # `len(targets) + len(stems)` stands for the mixture and S stems
            data = np.zeros((len(config['OUTPUTS']) + len(config['INPUTS']), 1,
                            valid_sample_length[i], config['N_CHANNELS']),
                            dtype=np.float32)  # (1 + S)xB=1xTxC

            data_key = list(data_valid[i])[0]
            audio_tags = list(data_valid[i])[0].split("-")
            stems = {}
            target = {}

            for k, tag in enumerate(audio_tags):
                
                if config['INPUTS'] == config['OUTPUTS']:
                    stems[tag] = data_valid[i][data_key]()[k][:valid_sample_length[i]]
                    target[tag] = data_valid[i][data_key]()[k][:valid_sample_length[i]]
                else:
                    if tag in config['INPUTS']:
                        stems[tag] = data_valid[i][data_key]()[k][:valid_sample_length[i]]
                    elif tag in config['OUTPUTS']:
                        target[tag] = data_valid[i][data_key]()[k][:valid_sample_length[i]]
                    
            

            for k, out in enumerate(config['OUTPUTS']):
                data[k] = target[out]

            for k, inp in enumerate(config['INPUTS']):
                data[k+len(config['OUTPUTS'])] = stems[inp]

            val_data.append(torch.from_numpy(data))

        with torch.no_grad():
            # move the validation data to the GPUs
            val_data = [v.to(f'cuda:{device_ids[i]}') for i, v in enumerate(val_data)]

            # forward pass
            val_outputs = nn.parallel.parallel_apply(replicas[:len(val_data)],
                                                     list(zip(val_data, val_losses)))

            # store loss results
            for idx, loss_dict in enumerate(val_outputs):
                for key in loss_dict:
                    valid_loss[key][data_indices[idx]] = loss_dict[key].item()

    # finally, launch the training/validation loop
    uprint(f'Starting training with {config["NUM_MINIBATCHES_PER_EPOCH"]} '
           f'minibatches per epoch ... ...')

    # we iterate over epochs:
    train_loss = np.zeros((config['NUM_EPOCHS'],))
    max_norm = np.zeros((config['NUM_EPOCHS'],))

    if config['GRAD_CLIP_MAX_NORM'] is None:
        # use adaptive gradient clipping scheme
        grad_history = np.zeros(config['NUM_EPOCHS'] * config['NUM_MINIBATCHES_PER_EPOCH'], dtype=np.float32)
        grad_history_idx = 0

        def _get_grad_norm(parameters, norm_type=float('inf')):
            """
            Get norm of the gradients.

            Args:
                parameters: parameters to compute the norm of gradients from
                norm_type: type of norm to compute

            Returns:
                total norm
            """
            parameters = [p for p in parameters if p.grad is not None]
            device = parameters[0].grad.device
            if norm_type == float('inf'):
                total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
            else:
                total_norm = torch.norm(torch.stack(
                    [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
            return total_norm.item()

    for epoch in range(config['NUM_EPOCHS']):
#     for epoch in range(1):
        timestamp1 = time.time()

        # set current learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate[epoch]
        if epoch in config['SAVE_NET_AT_EPOCHS']:
            torch.save(net.state_dict(),
                       results_folder + f'/current_model_for_{pretty_my_src}_after{str(epoch)}epochs.params')

        # accumulate loss/maximum gradient norm on GPU to avoid explicit synchronization
        _accum_loss = torch.zeros((), device=device_ids[0])
        _max_norm = torch.zeros((), device=device_ids[0])

#         # in each epoch, we do a full pass over the training data
        super_net.train()
        for minibatch in range(config['NUM_MINIBATCHES_PER_EPOCH']):

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            if config['QUANTIZATION_OP'] is not None:
                super_net.replace_with_quantized_weights()

            # get index of filled slot
            idx = queues_filled[my_src].get()

            # create variables
            _data = data_tensors[idx]

            # transfer the parameters to all GPUs
            replicas = nn.parallel.replicate(super_net, device_ids)

            # prepare data/loss
            data_scattered = nn.parallel.scatter(_data, device_ids, dim=1)
            loss_scattered = [{'train_loss': _} for _ in
                              nn.parallel.replicate(
                                config['TRAIN_LOSSES'][minibatch % len(config['TRAIN_LOSSES'])],
                                device_ids)
                              ]

#             # forward pass
            outputs = nn.parallel.parallel_apply(replicas, list(zip(data_scattered, loss_scattered)))

            # gather loss results
            _loss = torch.mean(nn.parallel.gather(outputs, device_ids[0])['train_loss'])

            # backward pass
            grad_scaler.scale(_loss).backward()

            # update statistics
            _accum_loss += _loss.detach()

            # tell data provider processes that this slot can be used again
            queues_empty[my_src].put(idx)

            grad_scaler.unscale_(optimizer)
            # scale gradient if it has too large values
            if config['GRAD_CLIP_MAX_NORM'] is None:
                # use adaptive scheme
                grad_history[grad_history_idx] = _get_grad_norm(net.parameters(), config['GRAD_CLIP_NORM_TYPE'])
                grad_history_idx += 1

                _max_norm = torch.maximum(_max_norm,
                                          nn.utils.clip_grad_norm_(net.parameters(),
                                                                   np.percentile(grad_history[:grad_history_idx], 10),
                                                                   config['GRAD_CLIP_NORM_TYPE']))
            else:
                # use fixed value
                _max_norm = torch.maximum(_max_norm,
                                          nn.utils.clip_grad_norm_(net.parameters(),
                                                                   config['GRAD_CLIP_MAX_NORM'],
                                                                   config['GRAD_CLIP_NORM_TYPE']))

            # update weights
            grad_scaler.step(optimizer)
            grad_scaler.update()

        # compute average training loss/maximum gradient norm
        train_loss[epoch] = _accum_loss.item() / config['NUM_MINIBATCHES_PER_EPOCH']
        max_norm[epoch] = _max_norm.item()

        timestamp2 = time.time()

        # perform validation
        super_net.eval()
        replicas = nn.parallel.replicate(super_net, device_ids)
        for i in range(0, len(data_valid), len(device_ids)):
            parallel_perform_valid(list(range(i, min(i+len(device_ids), len(data_valid)))), replicas)

        timestamp3 = time.time()

        # write train loss to tensorboard
        if config['TENSORBOARD']:
            writer_loss.add_scalar('train_loss', train_loss[epoch], epoch)
            writer_loss.flush()


            if config['TENSORBOARD'] == 'complete':
                for tag, parm in net.named_parameters():
                    writer_loss.add_histogram(tag, parm.grad.data.cpu().numpy(), state['epoch'])


        # plot training error curve (to jpeg image)
        if np.any(train_loss[:epoch + 1] <= 0):  # check for non-positive values (e.g., due to using `SDRLoss`)
            plt.plot(range(1, epoch + 2), train_loss[:epoch + 1], label='loss')

        else:
            plt.semilogy(range(1, epoch+2), train_loss[:epoch+1], label='loss')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.savefig(results_folder + '/loss_' + pretty_my_src + '.jpeg', bbox_inches='tight', dpi=300)
        plt.clf()

        # save current models
        fpartial = os.path.join(results_folder, f'current_model_for_{pretty_my_src}')
        torch.save(net.state_dict(), fpartial + '.params')

        if len(data_valid) > 0:
            # store validation error in each epoch as text file
            for key in valid_loss:
                with open(os.path.join(results_folder, f'VALID_ERROR_{key.upper()}_{pretty_my_src}.txt'), 'a') as f:
                    f.write(' '.join([f'{_:6.3f}' for _ in valid_loss[key]]))
                    f.write('\n')

            # save best model (on validation set)
            fpartial = os.path.join(results_folder, f'best_model_for_{pretty_my_src}_valid_')
            for key in valid_loss:
                mean_loss_this_epoch = np.mean(valid_loss[key])
                median_loss_this_epoch = np.median(valid_loss[key])

                # write validation losses to tensorboard
                if config['TENSORBOARD']:
                    writer_loss.add_scalar('validation_loss/' + key,
                                           mean_loss_this_epoch, epoch)
                    writer_loss.flush()

                if mean_loss_this_epoch < best_valid_mean[key]:
                    best_valid_mean[key] = mean_loss_this_epoch
                    torch.save(net.state_dict(), fpartial + key + '_mean.params')
                    if config['QUANTIZATION_OP'] is not None:
                        torch.save({_.replace('_float32', ''): recursive_getattr(net, _.replace('_float32', ''))
                                    for _ in net.state_dict()}, fpartial + key + '_mean.quant.params')
                if median_loss_this_epoch < best_valid_median[key]:
                    best_valid_median[key] = median_loss_this_epoch
                    torch.save(net.state_dict(), fpartial + key + '_median.params')
                    if config['QUANTIZATION_OP'] is not None:
                        torch.save({_.replace('_float32', ''): recursive_getattr(net, _.replace('_float32', ''))
                                    for _ in net.state_dict()}, fpartial + key + '_median.quant.params')

        # compute regularization strength (for monitoring)
        if config['L2_REGULARIZATION'] is None:
            weight_decay = 0.
        else:
            parameters = [p for p in net.parameters() if p.grad is not None]
            weight_decay = torch.norm(torch.stack([torch.norm(p.detach()) for p in parameters]))
            weight_decay = weight_decay.item()
            # square as reg is squared l2-norm
            weight_decay = config['L2_REGULARIZATION'] * np.square(weight_decay)

        timestamp4 = time.time()

        # print the results for this epoch
        train_time = timestamp2 - timestamp1
        valid_time = timestamp3 - timestamp2
        store_time = timestamp4 - timestamp3
        out_message = (f'{"["+pretty_my_src+"]":>10}'
                       f'\tEpoch {epoch+1} of {config["NUM_EPOCHS"]}'
                       f' took {train_time/60.:.2f}m + {valid_time/60:.2f}m + {store_time/60:.2f}m'
                       f' (finished in {(config["NUM_EPOCHS"]-epoch-1)*(timestamp4-timestamp1) / 3600:.1f}h)'
                       f'\ttrain={{loss: {train_loss[epoch]:.6f}, '
                       f'max-gradnorm: {max_norm[epoch]:.6f}, reg-term: {weight_decay:.6f}}}')
        if len(data_valid) > 0:
            out_message += '\tvalid={'
            for key in valid_loss:
                out_message += (f'{key} mean: {np.mean(valid_loss[key]):.6f}, '
                                f'{key} median: {np.median(valid_loss[key]):.6f}, ')
            out_message = out_message[:-2] + '}'
        uprint(out_message)

    # after finishing training: store training error values
    np.savez(results_folder + '/loss_' + pretty_my_src + '.npz', loss)

    # close validation thread pool
    tp.close()
    tp.join()

# Launch data providers (important: each data provider has his own random seed)
fill_queues_processes = []
for idx in range(config['NUM_DATAPROVIDING_PROCESSES']):
    seed1 = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)
    seed2 = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)
    uprint(f'Starting data providing process {idx} with random seed {seed1} and {seed2}')
    fill_queues_processes.append(Process(target=fill_queues, args=(seed1, seed2,)))
    fill_queues_processes[idx].daemon = True
    fill_queues_processes[idx].start()

# Launch training processes
train_network_processes = {}
for src in config['TARGETS']:
    train_network_processes[src] = Process(target=train_network,
                                           args=(src,))
    train_network_processes[src].daemon = True
    train_network_processes[src].start()

for src in config['TARGETS']:
    train_network_processes[src].join()

# terminate data providing processes
for p in fill_queues_processes:
    p.terminate()
