import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



aliases = np.load('np_data/snr_aliases_mesamini.npy')
harmonics = np.load('np_data/snr_harmonics_mesamini.npy')

midi = np.arange(21, 109)
f0_freqs = np.floor(440 * 2 ** ((midi - 69) / 12))
os_factors = [1, 48/44.1, 2, 96/44.1, 4, 196/44.1]
colors=['k', "#FF0000", "#00FF00", "#0000FF", "#00FFFF","#FF00FF"]


fig, axs = plt.subplots(figsize=[8, 4], nrows=1, ncols=2)
for o, os_factor in enumerate(os_factors):
    axs[0].semilogx(f0_freqs, harmonics[:, 0, 0, o], '--' if o == 0 else '-',
                    label='M = {}'.format(np.round(os_factor, 3)), color=colors[o])
axs[0].set_xlabel('Input fundamental frequency, f0 [Hz]')
axs[0].set_ylabel('SNRH [dB]', labelpad=-5)
axs[0].set_xlim([27.5, 4186])
axs[0].set_ylim([0, 130])
axs[0].grid(which='minor', alpha=0.1)
axs[0].grid(which='major', alpha=0.4)

#plt.figure(figsize=[8, 4])
for o, os_factor in enumerate(os_factors):
    axs[1].semilogx(f0_freqs, aliases[:, 0, 0, o], '--' if o == 0 else '-',
                    label='M={}'.format(np.round(os_factor, 2)), color=colors[o])
axs[1].set_xlabel('Input fundamental frequency, f0 [Hz]')
axs[1].set_ylabel('SNRA [dB]', labelpad=-5)
axs[1].set_xlim([27.5, 4186])
axs[1].set_ylim([0, 130])
axs[1].grid(which='minor', alpha=0.1)
axs[1].grid(which='major', alpha=0.4)
axs[0].legend(loc='lower left')
plt.subplots_adjust(wspace=0.2)

#plt.savefig('../figures/mesamini_os.pdf', bbox_inches='tight')
plt.show()