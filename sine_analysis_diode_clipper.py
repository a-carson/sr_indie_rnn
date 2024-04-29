import copy
import time
from sr_indie_rnn.modules import get_SRIndieDiodeClipper, LagrangeInterp_RNN, OptimalFIRInterp_RNN, APDL_RNN, STN_RNN
import torch
import numpy as np
import math
from sr_indie_rnn.utils import cheb_fft, bandlimited_harmonic_signal, get_odd_harmonics, snr_dB
from diode_clipper import DiodeClipper
from argparse import ArgumentParser, BooleanOptionalAction
import wandb

parser = ArgumentParser()
parser.add_argument("--sr_base", type=int, default=44100)
parser.add_argument("--sr_run", type=int, default=48000)
parser.add_argument("--gain", type=float, default=1.0)
parser.add_argument("--wandb", action=BooleanOptionalAction)
parser.add_argument('--project', type=str, default='sr_indie_diode_clipper')
parser.add_argument('-m', '--methods', nargs='+', default=['lagrange', 'opto'])
parser.add_argument('--fir_order', type=int, default=3)
parser.add_argument('--midi_min', type=int, default=21)
parser.add_argument('--midi_max', type=int, default=109)
parser.add_argument('--lut_path', type=str, default='diode_clipper_LUT_44.1kHz.npy')



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
    midi = torch.arange(args.midi_min, args.midi_max, dtype=dtype)

    # init
    f0_freqs = torch.floor(440 * 2 ** ((midi - 69) / 12))       # need int values
    snr_aliases = np.zeros((len(f0_freqs), len(methods)))
    snr_harmonics = np.zeros((len(f0_freqs), len(methods)))
    thd = np.zeros((len(f0_freqs)))

    start_time = time.time()

    # base time and input -------
    t_ax = torch.arange(0, math.ceil(sr_base * dur_seconds), dtype=dtype) / sr_base
    x = gain * torch.sin(2.0 * f0_freqs.view(-1, 1, 1) * torch.pi * t_ax.view(1, -1, 1))
    num_samples = math.ceil(sr_base * dur_seconds)

    print('SR ratio = ', os_factor)
    # oversampled time and input ---------
    sr = math.ceil(sr_base * os_factor)
    num_samples_up = math.ceil(sr * dur_seconds)
    t_ax_up = torch.arange(0, num_samples_up, dtype=dtype) / sr
    x_up = gain * torch.sin(2.0 * f0_freqs.view(-1, 1, 1) * torch.pi * t_ax_up.view(1, -1, 1))

    # process baseline output --------------------------
    base_model = DiodeClipper(sample_rate=sr_base, lut_path=lut_path)
    base_model.double()
    base_model.eval()

    for m, method in enumerate(methods):

        if args.wandb:
            run = wandb.init(project=args.project,
                       group='sr_base={}'.format(sr_base),
                       name='{}_{}'.format(method, order))

        # process modified model output ------------------
        model = copy.deepcopy(base_model)
        if method == 'lagrange':
            model.rec = LagrangeInterp_RNN(cell=base_model.rec.cell, order=order, os_factor=os_factor)
        elif method == 'opto':
            model.rec = OptimalFIRInterp_RNN(cell=base_model.rec.cell, order=order, os_factor=os_factor)
        elif method == 'apdl':
            model.rec = APDL_RNN(cell=base_model.rec.cell, os_factor=os_factor)
        elif method == 'stn':
            model.rec = STN_RNN(cell=base_model.rec.cell, os_factor=os_factor)
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
            freqs_base, amps_base, phase_base, dc_base = get_odd_harmonics(Y_base, f0, sr_base)
            y_base_bl = bandlimited_harmonic_signal(freqs_base, amps_base, phase_base, dc_base, t_ax, sr_base)

            # model bl harmonic sig
            freqs, amps, phase, dc = get_odd_harmonics(Y, f0, sr)
            num_harmonics = len(freqs_base)
            mask = (freqs < sr_base / 2).int()
            y_bl = bandlimited_harmonic_signal(freqs * mask,
                                               amps * mask,
                                               phase * mask,
                                                dc, t_ax, sr)

            # adjust to ensure equal first harmonic amps
            Y_bl = cheb_fft(y_bl)
            Y_bl *= Y[f0].abs() / Y_bl[f0].abs()

            # compute SNRs
            snra = snr_dB(sig=Y_bl[:num_samples//2].abs(),
                          noise=Y_bl[:num_samples//2].abs() - Y[:num_samples//2].abs())
            snr_aliases[i, m] = snra.numpy()

            snrh = snr_dB(sig=y_base_bl,
                          noise=y_bl - y_base_bl)
            snr_harmonics[i, m] = snrh.numpy()

            if args.wandb:
                run.log({"SNRAliasing_os={}".format(os_factor): snra.numpy(),
                           "SNRHarmonics_os={}".format(os_factor): snrh.numpy(),
                           "f0": f0})
        if args.wandb:
            run.finish()



        # plt.figure()
        # plt.semilogx(f0_freqs, snr_aliases[..., o])
        # plt.title('Aliasing SNR -- OS = {}'.format(os_factor))
        # plt.xlabel('f0 [Hz]'), plt.ylabel('dB')
        # #plt.ylim([0, 120])
        # plt.legend(methods)
        # if args.wandb:
        #     wandb.log({'SNRA': plt})
        # else:
        #     plt.show()
        #
        # plt.figure()
        # plt.semilogx(f0_freqs.numpy(), snr_harmonics[..., o])
        # plt.title('OS = {}'.format(os_factor))
        # plt.xlabel('f0 [Hz]'), plt.ylabel('SNHR [dB]')
        # #plt.ylim([0, 120])
        # plt.legend(methods)
        # if args.wandb:
        #     wandb.log({'SNRH': plt})
        # else:
        #     plt.show()

    print("elapsed time: ", time.time() - start_time)
    np.save('snra_sr={}.npy'.format(sr_base), snr_aliases)
    np.save('snrh_sr={}.npy'.format(sr_base), snr_harmonics)