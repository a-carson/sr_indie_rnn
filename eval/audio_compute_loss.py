import numpy as np
import torch
import torchaudio
import os
from sr_indie_rnn.giant_fft_resample import giant_fft_downsample


dir = '../audio/proteus'
filenames = os.listdir(dir)
filenames.sort()
filenames = np.load('np_data/model_names.npy')

methods = ['naive', 'stn', 'lidl', 'apdl', 'lagrange']
os_nums = [1, 160, 2, 320]
os_denoms = [1, 147, 1, 147]
base_rate = 44100

esr_array = np.zeros((len(filenames), len(os_nums), len(methods)))
fft_mae_array = np.zeros((len(filenames), len(os_nums), len(methods)))
spec_error = np.zeros((len(filenames), len(os_nums), len(methods)))
batch_length = 5

for i, f in enumerate(filenames):

    target, base_rate_check = torchaudio.load(os.path.join(dir, '{}_{}_{}.wav'.format(f, 44100, 'naive')))

    for k, (p, q) in enumerate(zip(os_nums, os_denoms)):
        for l, method in enumerate(methods):

            pred, sample_rate = torchaudio.load(os.path.join(dir,'{}_{}_{}.wav'.format(f, int(np.round(base_rate * p / q)), method)))

            assert(sample_rate == int(np.round(base_rate * p / q)))

            # METHOD 1 -- compare DOWN
            pred_down = giant_fft_downsample(pred, orig_freq=p, new_freq=q)
            esr_array[i, k, l] = torch.sum((pred_down - target)**2) / torch.sum(target ** 2)

            # METHOD 2 -- compare FFTs
            N = target.shape[-1]
            Np = pred.shape[-1]
            target_fft = torch.fft.fft(target, N) / N
            pred_fft = torch.fft.fft(pred, Np) / Np

            fft_loss = torch.sum(
                (torch.abs(target_fft[:, :N//2] - pred_fft[:, :N//2])) ** 2
            ) / torch.sum(torch.abs(target_fft[:, :N//2]) ** 2)

            print(20 * torch.log10(fft_loss))
            fft_mae_array[i, k, l] = fft_loss

#np.save('np_data/audio_esr.npy', esr_array)
esr_dB = 10 * np.log10(esr_array + 1e-12)
fft_dB = 10 * np.log10(fft_mae_array + 1e-12)
print(esr_dB)
