from sr_indie_rnn.modules import STNRNN, DelayLineRNN, HybridSTNDelayLineRNN, ADAARNN, AllPassDelayLineRNN, LagrangeDelayLineRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows
from utils import model_from_json
import librosa
import os

#mpl.use("macosx")

def my_fft(x, N=None):
    N_win = x.shape[-1]
    win = windows.chebwin(N_win, at=-120, sym=False)
    if N is None:
        N = N_win
    return np.fft.fft(x * win, n=N)

def get_harmonics(Y, f0, Fs):
    L = len(Y)
    fax = Fs * np.arange(0, len(Y)) / len(Y)
    harmonic_freqs = fax[f0: L//2: f0]
    harmonic_amps = np.abs(Y[f0: L//2: f0])
    harmonic_phase = np.angle(Y[f0: L//2: f0])
    dc_bin = np.real(Y[0])
    return harmonic_freqs, harmonic_amps, harmonic_phase, dc_bin

def bandlimited_harmonic_signal(freqs, amps, phase, DC_amp, t_ax, Fs):

    fourier_synth = DC_amp + 2 * torch.sum(
        torch.Tensor(amps).view(-1, 1) * torch.cos(
            2 * torch.pi * torch.Tensor(t_ax) * torch.Tensor(freqs).view(-1, 1) + torch.Tensor(phase).view(
                -1, 1)
        ), dim=0)

    fourier_synth *= 2 / Fs

    return fourier_synth.numpy()

def get_a_weighted_signal_and_bl(x, f0, Fs):
    L = len(x)
    time = torch.arange(0, L) / Fs
    f_ax = np.arange(0, L) / L * Fs
    a_weight = 10 ** (librosa.A_weighting(f_ax) / 10)


    X = my_fft(x)
    DC = np.real(X[0])

    freqs = f_ax[f0: L//2: f0]
    amps = np.abs(X[f0: L//2: f0])
    phases = np.angle(X[f0: L//2: f0])

    x_bl = DC + 2 * torch.sum(
        torch.from_numpy(amps).view(-1, 1) * torch.cos(
            2 * torch.pi * time * torch.from_numpy(freqs).view(-1, 1) + torch.from_numpy(phases).view(
                -1, 1)
        ), dim=0)

    x_bl *= 2 / Fs
    x_bl = x_bl.numpy()

    X_bl = my_fft(x_bl)
    X_bl *= np.abs(X[f0]) / np.abs(X_bl[f0])

    x_a_weight = np.real(np.fft.ifft(X * a_weight))
    x_bl_a_weight = np.real(np.fft.ifft(X_bl * a_weight))

    return x_a_weight, x_bl_a_weight


filenames = ['MesaMiniRec_HighGain_DirectOut.json']
base_path = '../../../Proteus_Tone_Packs/Selection'
os_factors = [1, 48/44.1, 2, 96/44.1, 4, 196/44.1]

# SETTINGS
methods = ['lagrange']
dur_seconds = 1.0
start_seconds = 0.0
sample_rate_base = 44100
gain = 0.1

midi = np.arange(21, 109)
f0_freqs = np.floor(440 * 2 ** ((midi - 69) / 12))

snr_aliases = np.zeros((len(f0_freqs), len(methods), len(filenames), len(os_factors)))
snr_harmonics = np.zeros((len(f0_freqs), len(methods), len(filenames), len(os_factors)))
thd = np.zeros((len(f0_freqs), len(filenames)))

for o, os_factor in enumerate(os_factors):
    print(os_factor)

    for f, filename in enumerate(filenames):
        print(filename)
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

        for m, method in enumerate(methods):

            if method == 'd_line':
                model.rec = DelayLineRNN(cell=cell)
            elif method == 'ap_d_line':
                model.rec = AllPassDelayLineRNN(cell=cell)
            elif method == 'hybrid':
                model.rec = HybridSTNDelayLineRNN(cell=cell)
            elif (method == 'stn') or (method == 'stn_improved'):
                model.rec = STNRNN(cell=cell)
                model.rec.improved = method == 'stn_improved'
            elif (method == 'adaa'):
                model.rec = ADAARNN(cell=cell)
            elif method == 'lagrange':
                model.rec = LagrangeDelayLineRNN(cell=cell)


            model.eval()
            with torch.no_grad():



                L = int(np.ceil(sample_rate_base * dur_seconds))
                t_ax = np.arange(0, L) / sample_rate_base
                x = gain * torch.sin(2.0 * torch.Tensor(f0_freqs).view(-1, 1, 1) * torch.pi * torch.Tensor(t_ax).view(1, -1, 1))
                model.reset_state()
                model.rec.os_factor = 1
                y_base, _ = model.forward(x)
                y_base = y_base[:, :, 0].detach().numpy()
                Y_base = my_fft(y_base)
                #plt.figure(), plt.plot(10 * np.log10(np.abs(Y_base[0, :]))), plt.show()
                f_ax_base = np.arange(0, L) / L * sample_rate_base
                a_weight_base = 10 ** (librosa.A_weighting(f_ax_base) / 10)

                sample_rate = np.round(sample_rate_base * os_factor)

                L_up = int(np.ceil(sample_rate * dur_seconds))
                t_ax_up = np.arange(0, L_up) / sample_rate
                x = gain * torch.sin(2.0 * torch.Tensor(f0_freqs).view(-1, 1, 1) * torch.pi * torch.Tensor(t_ax_up).view(1, -1, 1))
                model.reset_state()
                model.rec.os_factor = os_factor
                y, _ = model.forward(x)
                y = y[:, :, 0].detach().numpy()

                Y = my_fft(y)
                f_ax_up = np.arange(0, L_up) / L_up * sample_rate
                a_weight_up = 10 ** (librosa.A_weighting(f_ax_up) / 10)

                for i, f0 in enumerate(f0_freqs):
                    f0 = int(f0)
                    freqs, amps, phase, DC = get_harmonics(Y[i, ...], f0, sample_rate)
                    freqs_base, amps_base, phase_base, DC_base = get_harmonics(Y_base[i, ...], f0, sample_rate_base)
                    y_base_bl = bandlimited_harmonic_signal(freqs_base, amps_base, phase_base, DC_base, t_ax, sample_rate_base)

                    M = len(freqs_base)
                    y_down_bl = bandlimited_harmonic_signal(freqs[:M], amps[:M], phase[:M], DC, t_ax, sample_rate)

                    # NEW -- aliasing
                    Y_bl = np.abs(my_fft(y_down_bl))
                    Y_bl *= np.abs(Y[i, f0]) / np.abs(Y_bl[f0])
                    aliases = Y_bl[:L//2] - np.abs(Y[i, :L//2])
                    Y_bl = Y_bl[:L//2]

                    snr = 10 * np.log10(
                        np.sum(Y_bl ** 2) / (np.sum(aliases ** 2))
                    )
                    #print(snr)

                    esr = 10 * np.log10(
                        np.sum(y_base_bl ** 2) / (np.sum((y_down_bl - y_base_bl) ** 2))
                    )
                    snr_aliases[i, m, f, o] = snr
                    snr_harmonics[i, m, f, o] = esr
                    thd[i, f] = np.sqrt(np.sum(amps_base[1:]**2)) / amps_base[0]



    plt.figure(figsize=[10, 5])
    plt.semilogx(f0_freqs, np.mean(snr_aliases[..., o], axis=-1), marker='+')
    plt.title('Aliasing SNR -- OS = {}'.format(os_factor))
    plt.xlabel('f0 [Hz]'), plt.ylabel('dB')
    plt.legend(methods)

    plt.figure(figsize=[10, 5])
    plt.semilogx(f0_freqs, np.mean(snr_harmonics[..., o], axis=-1), marker='+')
    plt.title('OS = {}'.format(os_factor))
    plt.xlabel('f0 [Hz]'), plt.ylabel('SNHR [dB]')
    plt.legend(methods)
    plt.show()

#np.save('snr_aliases_mesamini.npy', snr_aliases)
#np.save('snr_harmonics_mesamini.npy', snr_harmonics)