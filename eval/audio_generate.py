import time

from sr_indie_rnn.modules import STNRNN, DelayLineRNN, HybridSTNDelayLineRNN, AllPassDelayLineRNN, LagrangeDelayLineRNN
import torch
import matplotlib as mpl
import torchaudio
from utils import model_from_json
import os
import pathlib
import numpy as np
from sr_indie_rnn.giant_fft_resample import giant_fft_upsample_torch

mpl.use("macosx")


filenames = [
            '6505Plus_Red_DirectOut.json',
            'MesaIICplus_Drive8_5EQoff.json',
            'BlackstarHT40_AmpClean.json',		'MesaMiniRec_HighGain_DirectOut.json',
            'BlackstarHT40_AmpHighGain.json',	'PlexiBreed_JTM_pedal.json',
            'BossMT2_PedalHighGain.json',   	'PrincetonAmp_Clean.json',
            'DumbleKit_HighG_DirectOut.json',	'ProcoRatPedal_HighGain.json',
            'DumbleKit_LowG_DirectOut.json',	'RockmanAcoustic_Pedal.json',
            'GoatPedal_HighGain.json',      	'RockmanXPR_HighGain.json',
            'LittleBigMuff_HighGainPedal.json', 'Splawn_OD_FractalFM3_HighGain.json',
            'MatchlessSC30_Ch1_DirectOut.json',	'XComp_Pedal.json']


base_path = '../../../Proteus_Tone_Packs/Selection'

# SETTINGS
methods = ['apdl']
#methods = ['naive']
dur_seconds = 1.0
start_seconds = 0.0
ps = [320]
qs = [147]
batch_length = 5.0
dur_zeros = 0.1
i = 0
base_sample_rate = 44100

data_path = '../../../audio_datasets/dist_fx_192k/44k/test/ht1-input.wav'

save_path = '../audio/proteus/'
start_time = time.time()
total_iterations = len(filenames) * len(ps) * len(methods)

print('Total iterations: ')

x, sample_rate_base = torchaudio.load(data_path)
x = x[...,:-1]

for f, filename in enumerate(filenames):

    device_name = pathlib.Path(filename).stem
    model = model_from_json.RNN_from_state_dict(os.path.join(base_path, filename))

    base_rnn = model.rec
    if type(model.rec) == torch.nn.LSTM:
        cell = torch.nn.LSTMCell(hidden_size=base_rnn.hidden_size,
                                 input_size=base_rnn.input_size,
                                 bias=base_rnn.bias)
    elif type(model.rec) == torch.nn.GRU:
        cell = torch.nn.GRUCell(hidden_size=base_rnn.hidden_size,
                                input_size=base_rnn.input_size,
                                bias=base_rnn.bias)
    else:
        cell = torch.nn.RNNCell(hidden_size=base_rnn.hidden_size,
                                input_size=base_rnn.input_size,
                                bias=base_rnn.bias)
    cell.weight_hh = base_rnn.weight_hh_l0
    cell.weight_ih = base_rnn.weight_ih_l0
    cell.bias_hh = base_rnn.bias_hh_l0
    cell.bias_ih = base_rnn.bias_ih_l0


    for p, q in zip(ps, qs):

        x_up = giant_fft_upsample_torch(x, orig_freq=q, new_freq=p)
        os_factor = p / q
        sample_rate = int(np.round(sample_rate_base * os_factor))

        batch_size = int(x_up.shape[-1] / batch_length / sample_rate)
        x_up = torch.reshape(x_up, (batch_size, x_up.shape[-1] // batch_size, 1))

        pad_samples = int(dur_zeros * sample_rate)
        pad = torch.zeros(batch_size, pad_samples, 1)

        for m, method in enumerate(methods):
            i += 1

            if method == 'lidl':
                model.rec = DelayLineRNN(cell=cell)
            elif method == 'apdl':
                model.rec = AllPassDelayLineRNN(cell=cell)
            elif method == 'hybrid':
                model.rec = HybridSTNDelayLineRNN(cell=cell)
            elif method == 'stn':
                model.rec = STNRNN(cell=cell)
                model.rec.improved = False
            elif method == 'lagrange':
                model.rec = LagrangeDelayLineRNN(cell=cell)


            model.eval()

            print('{}/{}: {} - os={} - {}'.format(i, total_iterations, device_name, os_factor, method))

            last_time = time.time()
            with torch.no_grad():
                model.reset_state()
                model.rec.os_factor = sample_rate / sample_rate_base
                y_up, _ = model(torch.cat((pad, x_up), dim=1))

                # remove pad and last dim
                y_up = torch.flatten(y_up[:, pad_samples:, :]).view(1, -1)

            torchaudio.save(os.path.join(save_path,
                                         '{}_{}_{}.wav'.format(device_name, sample_rate, method)),
                            y_up,
                            channels_first=True,
                            sample_rate=sample_rate,
                            bits_per_sample=32)


            print('Iteration time: {}. Elapsed time: {}'.format(
                time.time() - last_time,
                time.time() - start_time))







