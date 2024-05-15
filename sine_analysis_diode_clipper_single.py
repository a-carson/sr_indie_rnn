import torch
import math
from diode_clipper import DiodeClipper
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from sr_indie_rnn.modules import get_SRIndieDiodeClipper, LagrangeInterp_RNN, OptimalFIRInterp_RNN, APDL_RNN, STN_RNN
from sr_indie_rnn.utils import cheb_fft, get_odd_harmonics
import numpy as np

mpl.use('macosx')

# settings
f0 = 69
sr_base = 48000
sr_run = 44100
os_factor = np.double(sr_run) / np.double(sr_base)
dur_seconds = 1.0
dtype = torch.double
gain = 1.0
methods = [
    {'filter': 'lagrange', 'order': 5},
    {'filter': 'opto', 'order': 5},
           ]
lut_path = 'diode_clipper_LUT_48kHz.npy'
#lut_path = None
# base time and input -------
num_samples = math.ceil(sr_base * dur_seconds)
t_ax = torch.arange(0, num_samples, dtype=dtype) / sr_base
f_ax = sr_base * torch.arange(0, num_samples, dtype=dtype) / num_samples
x = gain * torch.sin(2.0 * f0 * torch.pi * t_ax.view(1, -1, 1))
base_model = DiodeClipper(sample_rate=sr_base, lut_path=lut_path)
base_model.double()
y, _ = base_model(x)
y = y.flatten()
Y = cheb_fft(y)
freqs_base, amps_base, phase_base, dc_base = get_odd_harmonics(Y, f0, sr_base)

# up -------------
sr = math.ceil(sr_base * os_factor)
num_samples_up = math.ceil(sr * dur_seconds)
t_ax_up = torch.arange(0, num_samples_up, dtype=dtype) / sr
x_up = gain * torch.sin(2.0 * f0 * torch.pi * t_ax_up.view(1, -1, 1))
f_ax_up = sr * torch.arange(0, num_samples_up, dtype=dtype) / num_samples_up

plt.figure(1)
plt.plot(t_ax, y.flatten().detach(), label='base', color='k')
plt.figure(2)
plt.plot(f_ax, 10 * torch.log10(Y.flatten().abs()), label='base', color='k', linewidth=0.25)
plt.plot(freqs_base, 10 * torch.log10(amps_base), color='k')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c21']


for m, method in enumerate(methods):

    # process modified model output ------------------
    model = copy.deepcopy(base_model)
    if method['filter'] == 'lagrange':
        model.rec = LagrangeInterp_RNN(cell=base_model.rec.cell, order=method['order'], os_factor=os_factor)
    elif method['filter'] == 'opto':
        model.rec = OptimalFIRInterp_RNN(cell=base_model.rec.cell, order=method['order'], os_factor=os_factor)
    elif method['filter'] == 'apdl':
        model.rec = APDL_RNN(cell=base_model.rec.cell, os_factor=os_factor)
    elif method['filter'] == 'stn':
        model.rec = STN_RNN(cell=base_model.rec.cell, os_factor=os_factor)
    elif method['filter'] == 'exact':
        model = DiodeClipper(sample_rate=sr, lut_path=lut_path)

    model.eval()
    model.double()
    model.rec.os_factor = os_factor

    y_up, _ = model(x_up)
    y_up = y_up.flatten().detach()
    plt.figure(1)
    plt.plot(t_ax_up, y_up, label=method)

    Y_up = cheb_fft(y_up) / os_factor
    plt.figure(2)
   # plt.plot(f_ax_up, 10 * torch.log10(Y_up.abs()), label='{}_{}'.format(method, order), linewidth=0.25)
    freqs, amps, _, _ = get_odd_harmonics(Y_up, f0, sr)
    plt.plot(freqs, 10 * torch.log10(amps), '--', linewidth=0.75, label='{}_{}'.format(method['filter'], method['order']))

plt.figure(2)
plt.ylim([-20, 40])
plt.xlim([0, sr_base/2])
plt.legend()
plt.show()

