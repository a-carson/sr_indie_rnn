import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


names = np.load('np_data/model_names_short.npy')
aliases = np.load('np_data/snr_aliases_OS=1.0884353741496597.npy')
harmonics = np.load('np_data/snr_harmonics_OS=1.0884353741496597.npy')
thd = np.load('np_data/thd.npy')

#### SORT
idx = np.argsort(names)
names = names[idx]
aliases = aliases[..., idx]
harmonics = harmonics[..., idx]

methods = ['Naive', 'STN', 'LIDL', 'APDL', 'CIDL']
midi = np.arange(21, 109)
f0_freqs = np.floor(440 * 2 ** ((midi - 69)/12))

num_rows = 5
num_cols = 4
fontsize = 10
xpad = -1
ypad = -10

fig1, axs1 = plt.subplots(num_rows, num_cols, figsize=[15, 15*1.2/3*2*.85])
#plt.suptitle('Aliasing SNR -- M=1.0884')
fig2, axs2 = plt.subplots(num_rows, num_cols, figsize=[15, 15*1.2/3*2*.85])
#plt.suptitle('Harmonic SNR -- M=1.0884')


for i, f in enumerate(names):
    row = int(np.floor(i / num_cols))
    col = np.mod(i,  num_cols)
    axs1[row, col].semilogx(f0_freqs, aliases[..., i])
    axs1[row, col].set_title('{}) {}'.format(chr(i + 97), f), fontsize=fontsize)
    axs1[row, col].set_xlim([27.5, 4186])
    axs1[row, col].set_ylim([-10, 125])
    axs1[row, col].set_xlabel('f0 [Hz]', labelpad=ypad, fontsize=fontsize)
    axs1[row, col].set_ylabel('SNRA [dB]', labelpad=xpad, fontsize=fontsize)
    axs1[row, col].grid(alpha=0.4, which='major')
    axs1[row, col].grid(alpha=0.1, which='minor')
    axs2[row, col].semilogx(f0_freqs, harmonics[..., i])
    axs2[row, col].set_title('{}) {}'.format(chr(i + 97), f), fontsize=fontsize)
    axs2[row, col].set_xlim([27.5, 4186])
    axs2[row, col].set_ylim([-10, 125])
    axs2[row, col].set_xlabel('f0 [Hz]', labelpad=ypad, fontsize=fontsize)
    axs2[row, col].set_ylabel('SNRH [dB]', labelpad=xpad, fontsize=fontsize)
    axs2[row, col].grid(alpha=0.4, which='major')
    axs2[row, col].grid(alpha=0.1, which='minor')


axs1[-1, -2].set_axis_off(), axs1[-1, -1].set_axis_off()
axs2[-1, -2].set_axis_off(), axs2[-1, -1].set_axis_off()
axs1[-1, -3].legend(methods, bbox_to_anchor=(1.2, 1))
axs2[-1, -3].legend(methods, bbox_to_anchor=(1.2, 1))
fig1.subplots_adjust(hspace=0.5, wspace=0.25)
fig2.subplots_adjust(hspace=0.5, wspace=0.25)

#fig1.savefig('../figures/sine_aliasing_snr_48k_5x4.pdf', bbox_inches='tight')
#fig2.savefig('../figures/sine_harmonic_snr_48k_5x4.pdf', bbox_inches='tight')

plt.show()