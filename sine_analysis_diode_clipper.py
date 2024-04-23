import copy
import time
from sr_indie_rnn.modules import get_SRIndieDiodeClipper
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from sr_indie_rnn.utils import cheb_fft, bandlimited_harmonic_signal, get_harmonics, snr_dB
from diode_clipper import DiodeClipper
from argparse import ArgumentParser, BooleanOptionalAction
import wandb

parser = ArgumentParser()
parser.add_argument("--sr_base", type=int, default=44100)
parser.add_argument("--gain", type=float, default=1.0)
parser.add_argument("--wandb", action=BooleanOptionalAction)
parser.add_argument('--project', type=str, default='sr_indie_diode_clipper')
parser.add_argument('-m', '--methods', nargs='+', default=['naive', 'stn', 'lidl', 'apdl', 'cidl', 'lagrange_5', 'exact'])
parser.add_argument('--midi_min', type=int, default=21)
parser.add_argument('--midi_max', type=int, default=109)



if __name__ == '__main__':

    args = parser.parse_args()
    methods = args.methods

    dur_seconds = 1.0
    sr_base = args.sr_base
    if sr_base == 44100:
        os_factors = np.array([48 / 44.1, 2, 96/44.1], dtype=np.double)
    elif sr_base == 48000:
        os_factors = np.array([44.1 / 48, 2, 88.2/48], dtype=np.double)
    elif sr_base == 88200:
        os_factors = np.array([0.5, 48/88.2, 96/88.2], dtype=np.double)
    elif sr_base == 96000:
        os_factors = np.array([44.1/96, 0.5, 88.2/96], dtype=np.double)


    gain = args.gain
    dtype = torch.double
    midi = torch.arange(args.midi_min, args.midi_max, dtype=dtype)

    # init
    f0_freqs = torch.floor(440 * 2 ** ((midi - 69) / 12))       # need int values
    snr_aliases = np.zeros((len(f0_freqs), len(methods), len(os_factors)))
    snr_harmonics = np.zeros((len(f0_freqs), len(methods), len(os_factors)))
    thd = np.zeros((len(f0_freqs)))

    start_time = time.time()

    # base time and input -------
    t_ax = torch.arange(0, math.ceil(sr_base * dur_seconds), dtype=dtype) / sr_base
    x = gain * torch.sin(2.0 * f0_freqs.view(-1, 1, 1) * torch.pi * t_ax.view(1, -1, 1))
    num_samples = math.ceil(sr_base * dur_seconds)


    for o, os_factor in enumerate(os_factors):

        print(os_factor)
        # oversampled time and input ---------
        sr = math.ceil(sr_base * os_factor)
        num_samples_up = math.ceil(sr * dur_seconds)
        t_ax_up = torch.arange(0, num_samples_up) / sr
        x_up = gain * torch.sin(2.0 * f0_freqs.view(-1, 1, 1) * torch.pi * t_ax_up.view(1, -1, 1))

        # process baseline output --------------------------
        base_model = DiodeClipper(sample_rate=sr_base)
        base_model.double()
        base_model.eval()

        for m, method in enumerate(methods):

            if args.wandb:
                run = wandb.init(project=args.project,
                           group='sr_base={}'.format(sr_base),
                           name=method)

            # process modified model output ------------------
            if method == 'exact':
                model = DiodeClipper(sample_rate=sr)
            elif method == 'naive':
                model = copy.deepcopy(base_model)
            else:
                model = get_SRIndieDiodeClipper(base_model=base_model, method=method)
            model.eval()
            model.double()
            model.rec.os_factor = os_factor

            for i, f0 in enumerate(f0_freqs):
                f0 = int(f0)
                print(f0)

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

                # compute SNRs
                snra = snr_dB(sig=Y_bl[:num_samples//2].abs(),
                              noise=Y_bl[:num_samples//2].abs() - Y[:num_samples//2].abs())
                snr_aliases[i, m, o] = snra.numpy()

                snrh = snr_dB(sig=y_base_bl,
                              noise=y_bl - y_base_bl)
                snr_harmonics[i, m, o] = snrh.numpy()

                if args.wandb:
                    run.log({"snra_os={}".format(os_factor): snra.numpy(),
                               "snra_h={}".format(os_factor): snrh.numpy(),
                               "f0": f0})

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