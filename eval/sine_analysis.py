import scipy.signal

from sr_indie_rnn.sr_indie_train import SRIndieRNN, BaselineRNN
from sr_indie_rnn.modules import STNRNN, DelayLineRNN, HybridSTNDelayLineRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp, spectrogram
import torchaudio
import matplotlib as mpl
mpl.use("macosx")

def my_fft(x, N=None):
    N_win = x.shape[-1]
    if N is None:
        N = N_win
    return np.fft.fft(x * np.hanning(N_win), n=N)
    #return np.fft.fft(x, n=N)

input_paths = ['../../../audio_datasets/dist_fx/test/muff-input.wav',
               '../../../audio_datasets/dist_fx/88kHz/test/muff-input.wav']
target_paths = ['../../../audio_datasets/dist_fx/test/muff-target.wav',
               '../../../audio_datasets/dist_fx/88kHz/test/muff-target.wav']

# SETTINGS
method = 'hybrid'
dur_seconds = 1.0
start_seconds = 40.0
OS = [1, 48/44.1, 2, 96/44.1]
sample_rate_base = 44100

in_type = 'sine'
gain = 0.1
f0 = 1245
resample_before_plot = False
cutoff_freq_snr = 10e3

model = BaselineRNN.load_from_checkpoint('../pretrained/ht1_gru64_epoch=898-step=830676.ckpt', map_location='cpu').model
base_rnn = model.rec
cell = torch.nn.GRUCell(hidden_size=base_rnn.hidden_size,
                        input_size=base_rnn.input_size,
                        bias=base_rnn.bias)
cell.weight_hh = base_rnn.weight_hh_l0
cell.weight_ih = base_rnn.weight_ih_l0
cell.bias_hh = base_rnn.bias_hh_l0
cell.bias_ih = base_rnn.bias_ih_l0


if (method == 'd_line') or (method == 'stn') or (method == 'stn_improved') or (method == 'hybrid'):
    model = BaselineRNN.load_from_checkpoint('../pretrained/ht1_gru64_epoch=898-step=830676.ckpt', map_location='cpu').model
    if method == 'd_line':
        child_rec = DelayLineRNN(cell=cell)
    elif method == 'hybrid':
        child_rec = HybridSTNDelayLineRNN(cell=cell)
    else:
        child_rec = STNRNN(cell=cell)
        child_rec.improved = method == 'stn_improved'
    # copy weights to variable SR model
    model.rec = child_rec
elif method == 'conditioned':
    model = SRIndieRNN.load_from_checkpoint('../pretrained/muff_sr_indie_gru16.ckpt', map_location='cpu').model
else:
    model = BaselineRNN.load_from_checkpoint('../pretrained/muff_improved_slow_gru_16_pretrained.ckpt',
                                             map_location='cpu').model

model.rec.improved = method == 'stn_improved'

colors = ['k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c21']
plt.figure(figsize=[10, 5])
plt.figure(figsize=[10, 5])
model.eval()
with torch.no_grad():

    for i, os in enumerate(OS):
        sample_rate = sample_rate_base * os
        L = int(np.ceil(sample_rate * dur_seconds))
        t_ax = np.arange(0, L) / sample_rate
        if in_type == 'sine':
            x = gain * torch.sin(2.0 * f0 * torch.pi * torch.Tensor(t_ax)).reshape(1, L, 1)
        elif in_type == 'chirp':
            x = gain * torch.Tensor(chirp(t_ax, t1=dur_seconds, f0=0.0, f1=10e3)).view(1, L, 1)
        elif in_type == 'impulse':
            x = gain * torch.cat((torch.zeros(1, L//2, 1), torch.ones(1, 1, 1), torch.zeros(1, L//2-1, 1)), dim=1)
        elif in_type == 'audio':
            S = int(start_seconds * sample_rate)
            x, sample_rate = torchaudio.load(input_paths[i])
            y_target, _ = torchaudio.load(target_paths[i])
            assert(_ == sample_rate)
            x = x[..., S:S+L].view(1, L, 1)
            y_target = y_target[..., S:S+L].view(1, L, 1)
        else:
            x = gain * (torch.rand(1, L, 1) - 0.5)
        # process baselines
        model.reset_state()
        model.rec.os_factor = os
        if method == 'conditioned':
            y, _ = model.forward(
                torch.cat((x, os * torch.ones_like(x)), dim=-1)
            )
        else:
            y, _ = model.forward(x)

        y = y[0, :, 0].detach()
        #y = torch.roll(y, 1-os)


        if in_type == 'chirp':
            y = torchaudio.functional.resample(y, orig_freq=os, new_freq=1).numpy()
            plt.figure(i)
            f, t, Sxx = spectrogram(y, sample_rate_base, nfft=4096, window='hamming')
            plt.pcolormesh(t, f, 20 * np.log10(Sxx), shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('OS = {}'.format(os))
            plt.show()
        elif in_type == 'audio':
            plt.figure(2)
            plt.plot(t_ax, torch.flatten(y_target), label='Fs = {:.1f}kHz (target)'.format(sample_rate / 1000))
            esr = torch.mean((y - y_target)**2)
            print(esr.numpy())
        elif in_type == 'sine':

            if resample_before_plot:
                y = torchaudio.functional.resample(y, orig_freq=os, new_freq=1)
                os = 1
            y = y.numpy()

            # Plot frequency
            Nfft = len(y)
            f_ax = sample_rate_base * os * np.arange(0, Nfft) / Nfft
            Y = my_fft(y)/os

            # plot spectra
            plt.figure(1)
            plt.plot(f_ax, 10 * np.log10(np.abs(Y)), linewidth=1.0, color=colors[i])

            if in_type == 'sine':

                harmonic_freqs = f_ax[f0:int(Nfft/2):int(f0)]
                harmonic_amps = np.abs(Y[f0:int(Nfft/2):int(f0)])
                harmonic_phase = np.angle(Y[f0:int(Nfft/2):int(f0)])
                t_ax = torch.arange(0, len(y)) / (sample_rate_base * os)
                DC = np.abs(Y[0]) * np.sign(np.mean(y))
                fourier_synth = (torch.sum(
                    torch.from_numpy(harmonic_amps).view(-1, 1) * torch.cos(
                        2 * torch.pi * t_ax * torch.from_numpy(harmonic_freqs).view(-1, 1) + torch.from_numpy(harmonic_phase).view(-1, 1)
                    ), dim=0) * 2 + DC).numpy() / sample_rate_base

                fourier_synth *= 2


                # low-pass for SNR calculation only
                y = torchaudio.functional.lowpass_biquad(torch.from_numpy(y), sample_rate=sample_rate_base * os, cutoff_freq=cutoff_freq_snr).numpy()
                fourier_synth = torchaudio.functional.lowpass_biquad(torch.from_numpy(fourier_synth), sample_rate=sample_rate_base * os, cutoff_freq=cutoff_freq_snr).numpy()


                aliases = fourier_synth - y
                snr = 10 * np.log10(
                    np.sum(fourier_synth[Nfft//2:] ** 2) / (np.sum(aliases[Nfft//2:] ** 2))
                )

                # sanity check for fourier synth signal
                # plt.plot(f_ax, 10 * np.log10(np.abs(my_fft(fourier_synth)/os)), linewidth=1.0, color='k'), plt.show()
                plt.plot(harmonic_freqs, 10 * np.log10(harmonic_amps),
                         color=colors[i],
                         linestyle='-' if i == 0 else '--',
                         label='Fs = {:.1f}kHz, SNR={:.1f}dB'.format(sample_rate / 1000, snr))

        # plot waveform
        plt.figure(2)
        esr = 0.0
        plt.plot(t_ax, y, label='Fs = {:.1f}kHz (model), ESR={:.3f}%'.format(sample_rate / 1000, esr*100))



plt.figure(1)
plt.title('{} method -- {} input'.format(method, in_type))

plt.xlabel('f [Hz]')
plt.ylabel('[dB]')
plt.xlim([0, 20e3])
plt.ylim(([-30, 40]))
plt.legend(loc='lower left')

plt.figure(2)
plt.title('{} method -- {} input'.format(method, in_type))
plt.xlabel('t')
plt.ylabel('y')
#plt.xlim([0, 20e3])
#plt.ylim(([-30, 40]))
plt.legend(loc='lower left')
plt.show()

print('Done')