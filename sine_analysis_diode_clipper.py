import copy
import time
import sr_indie_rnn.modules as srirnn
import torch
import numpy as np
import math
from sr_indie_rnn.utils import cheb_fft, bandlimited_harmonic_signal, get_odd_harmonics, snr_dB, get_harmonics
from diode_clipper import DiodeClipper
from argparse import ArgumentParser, BooleanOptionalAction
import wandb
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('macosx')

parser = ArgumentParser()
parser.add_argument("--sr_base", type=int, default=44100)
parser.add_argument("--sr_run", type=int, default=48000)
parser.add_argument("--gain", type=float, default=1.0)
parser.add_argument("--wandb", action=BooleanOptionalAction)
parser.add_argument('--project', type=str, default='sr_indie_diode_clipper')
parser.add_argument('--group_name', type=str, default='')
parser.add_argument('-m', '--methods', nargs='+', default=['lagrange', 'opto', 'opto_dc'])
parser.add_argument('--fir_order', type=int, default=3)
parser.add_argument('--midi_min', type=int, default=21)
parser.add_argument('--midi_max', type=int, default=109)
parser.add_argument('--midi_step', type=int, default=1)
parser.add_argument('--lut_path', type=str, default='diode_clipper_LUT_44.1kHz.npy')
parser.add_argument('--model_path', type=str, default='')


if __name__ == '__main__':

    args = parser.parse_args()
    methods = args.methods
    if args.lut_path == '':
        lut_path = None
    else:
        lut_path = args.lut_path

    dur_seconds = 1.0
    sr_base = args.sr_base
    sr = args.sr_run
    os_factor = np.double(sr) / np.double(sr_base)
    order = args.fir_order

    gain = args.gain
    dtype = torch.double
    midi = torch.arange(args.midi_min, args.midi_max, args.midi_step, dtype=dtype)

    # init
    f0_freqs = torch.floor(440 * 2 ** ((midi - 69) / 12))       # need int values
    snr_aliases = np.zeros((len(f0_freqs), len(methods)))
    snr_spec = np.zeros((len(f0_freqs), len(methods)))
    snr_time = np.zeros((len(f0_freqs), len(methods)))

    thd = np.zeros((len(f0_freqs)))

    start_time = time.time()

    # base time and input -------
    t_ax = torch.arange(0, math.ceil(sr_base * dur_seconds), dtype=dtype) / sr_base
    x = gain * torch.sin(2.0 * f0_freqs.view(-1, 1, 1) * torch.pi * t_ax.view(1, -1, 1))
    num_samples = math.ceil(sr_base * dur_seconds)

    print('SR ratio = ', os_factor)
    # oversampled time and input ---------
    num_samples_up = sr
    t_ax_up = torch.arange(0, num_samples_up, dtype=dtype) / sr
    x_up = gain * torch.sin(2.0 * f0_freqs.view(-1, 1, 1) * torch.pi * t_ax_up.view(1, -1, 1))

    # process baseline output --------------------------
    if args.model_path == '':
        base_model = DiodeClipper(sample_rate=sr_base, lut_path=lut_path)
        base_model.double()
        base_model.eval()
        cell = base_model.rec.cell
        get_harmonics = get_odd_harmonics
    else:
        base_model = srirnn.get_AudioRNN_from_json(args.model_path)
        base_model.double()
        base_model.eval()
        cell = srirnn.get_cell_from_rnn(base_model.rec)


    for m, method in enumerate(methods):

        if args.wandb:
            if args.group_name == '':
                group_name = 'sr_base={}'.format(sr_base)
            else:
                group_name = args.group_name
            run = wandb.init(project=args.project,
                       group=group_name,
                       name='{}_{}'.format(method, order))

        # process modified model output ------------------
        model = copy.deepcopy(base_model)
        if method == 'lagrange':
            model.rec = srirnn.LagrangeInterp_RNN(cell=cell, order=order, os_factor=os_factor)
        elif method == 'opto':
            model.rec = srirnn.OptimalFIRInterp_RNN(cell=cell, order=order, os_factor=os_factor)
        elif method == 'opto_dc':
            model.rec = srirnn.OptimalFIRInterp_RNN(cell=cell, order=order, os_factor=os_factor, dc_flat=True)
        elif method == 'apdl':
            model.rec = srirnn.APDL_RNN(cell=cell, os_factor=os_factor)
        elif method == 'stn':
            model.rec = srirnn.STN_RNN(cell=cell, os_factor=os_factor)
        else:
            print('Naive method')
        print(model)
        model.eval()
        model.double()
        model.rec.os_factor = os_factor

        for i, f0 in enumerate(f0_freqs):
            f0 = int(f0)
            print('midi = {}; f0 = {}Hz'.format(midi[i], f0))

            with torch.no_grad():
                y_base, _ = base_model.forward(x[i, ...].unsqueeze(0))
                y_base = y_base[0, :, 0].detach()
                Y_base = cheb_fft(y_base)

                y, _ = model.forward(x_up[i, ...].unsqueeze(0))
                y = y[0, :, 0].detach()
                Y = cheb_fft(y)

            # baseline bl harmonic sig
            freqs_base, amps_base, phase_base, dc_base = get_harmonics(Y_base, f0, sr_base)
            y_base_bl = bandlimited_harmonic_signal(freqs_base, amps_base, phase_base, dc_base, t_ax, sr_base)

            # model bl harmonic sig
            freqs, amps, phase, dc = get_harmonics(Y, f0, sr)
            num_harmonics = len(freqs_base)
            mask = (freqs < sr_base / 2).int()
            y_bl = bandlimited_harmonic_signal(freqs * mask,
                                               amps * mask,
                                               phase * mask,
                                                dc, t_ax, sr)

            # adjust to ensure equal first harmonic amps
            Y_bl = cheb_fft(y_bl)
            Y_bl *= Y[f0].abs() / Y_bl[f0].abs()
            Y_base_bl = cheb_fft(y_base_bl)
            Y_base_bl *= Y_base[f0].abs() / Y_base_bl[f0].abs()


            # compute SNRs
            snra = snr_dB(sig=Y_bl[:num_samples//2].abs(),
                          noise=Y_bl[:num_samples//2].abs() - Y[:num_samples//2].abs())
            snr_aliases[i, m] = snra.numpy()
            print('SNRA =', snra.numpy())

            snrs = snr_dB(sig=Y_base_bl[:num_samples//2].abs() / sr_base,
                          noise=Y_bl[:num_samples//2].abs() / sr - Y_base_bl[:num_samples//2].abs() / sr_base)
            snr_spec[i, m] = snrs.numpy()
            print('SNR_Spec =', snrs.numpy())

            snrt = snr_dB(sig=y_base_bl,
                          noise=y_bl - y_base_bl)
            snr_time[i, m] = snrt.numpy()
            print('SNR_Time =', snrt.numpy())

            if args.wandb:
                run.log({"SNRAliasing_os={}".format(os_factor): snra.numpy(),
                           "SNR_Time_os={}".format(os_factor): snr_time.numpy(),
                           "f0": f0,
                           "midi_note": midi[i]})
        if args.wandb:
            run.finish()



    plt.figure()
    plt.semilogx(f0_freqs, snr_aliases)
    plt.title('Aliasing SNR -- OS = {}'.format(os_factor))
    plt.xlabel('f0 [Hz]'), plt.ylabel('dB')
    plt.ylim([0, 120])
    plt.legend(methods)
    plt.show()

    plt.figure()
    plt.semilogx(f0_freqs.numpy(), snr_spec)
    plt.title('OS = {}'.format(os_factor))
    plt.xlabel('f0 [Hz]'), plt.ylabel('SNR_Spec [dB]')
    plt.ylim([0, 120])
    plt.legend(methods)
    plt.show()

    plt.figure()
    plt.semilogx(f0_freqs.numpy(), snr_time)
    plt.title('OS = {}'.format(os_factor))
    plt.xlabel('f0 [Hz]'), plt.ylabel('SNR_Time [dB]')
    plt.ylim([0, 120])
    plt.legend(methods)
    plt.show()

    print("elapsed time: ", time.time() - start_time)
    np.save('snra_sr={}.npy'.format(sr_base), snr_aliases)
    np.save('snrs_sr={}.npy'.format(sr_base), snr_spec)
    np.save('snrt_sr={}.npy'.format(sr_base), snr_time)