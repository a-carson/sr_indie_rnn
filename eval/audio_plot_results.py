import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('macosx')

esr = np.load('audio_esr.npy')
thd = np.load('thd.npy').mean(0)

methods = ['Naive', 'STN', 'LIDL', 'APDL', 'CIDL']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']#, '#8c564b','#e377c21']
filenames = np.array([
            '6505Plus_Red_DirectOut', 'MesaIICplus_Drive8_5EQoff',
            'MesaMiniRec_HighGain_DirectOut', 'BlackstarHT40_AmpClean',
            'BlackstarHT40_AmpHighGain', 'PlexiBreed_JTM_pedal',
            'BossMT2_PedalHighGain', 'PrincetonAmp_Clean',
            'DumbleKit_HighG_DirectOut', 'ProcoRatPedal_HighGain',
            'DumbleKit_LowG_DirectOut',	'RockmanAcoustic_Pedal',
            'GoatPedal_HighGain', 'RockmanXPR_HighGain',
            'LittleBigMuff_HighGainPedal', 'Splawn_OD_FractalFM3_HighGain',
            'MatchlessSC30_Ch1_DirectOut', 'XComp_Pedal'])

#
filenames = np.array([
            '6505Plus', 'MesaIICplus',
            'MesaMiniRec', 'HT40_Clean',
            'HT40_HighG', 'PlexiBreed_JTM',
            'BossMT2', 'PrincetonAmp',
            'Dumble_HighG', 'ProcoRat',
            'Dumble_LowG',  'RockmanAc.',
            'GoatPedal', 'RockmanXPR',
            'LittleBigMuff', 'Splawn',
            'MatchlessSC30', 'XComp'])


#### SORT
idx = np.argsort(filenames)
filenames = filenames[idx]
esr = esr[idx, ...]

#filenames = filenames[:10]

snr = -10 * np.log10(esr)

SRs = [48, 88.2, 96]
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=[15, 15*1.3/3*.85])

for s, sr in enumerate(SRs):

    data = snr[:, s+1, :]

    # create dat
    x = np.arange(len(filenames))
    width = 0.2
    #fig, ax = plt.subplots(figsize=[15, 3])


    # Set the number of groups (N) and the number of colors (M)
    N = len(filenames)
    M = len(methods)

    # Set up the figure and axis
    # Set the width of the bars based on the number of colors (M)
    bar_width = 0.2
    group_spacing = 1.5
    #fig, axs = plt.subplots(N, 1, figsize=[16, 20])

    # Plot the bars for each group
    # for i in range(N):
    #     bars = ax.barh((np.arange(M)*N + i)*bar_width, data[i, :], height=bar_width, color=colors)

    for m in range(M):
        for n in range(N):
            bars = ax[s].bar((n*M*group_spacing + m)*bar_width, data[n, m], width=bar_width, color=colors[m])

    # Set labels and title
    ax[s].set_ylabel('SNR [dB]')
    ax[s].set_title('{}) M = {}x oversampling (44.1kHz -> {}kHz)'.format(chr(s+97), np.round(sr/44.1, 3), sr), fontsize=10)

    # Set y-axis ticks and labels
    ax[s].set_xticks(np.arange(N) * group_spacing + 2*bar_width)
    ax[s].set_xticklabels('')
    ax[s].set_ylim([-5, 70])
    ax[s].set_yticks(np.arange(0, 8) * 10)
    ax[s].set_xlim([-bar_width, 26.5])


    # Show the plot
    ax[s].grid(alpha=0.1, which='both')
    #plt.savefig('../figures/audio_snr_{}k.pdf'.format(sr),bbox_inches='tight')

# Add a legend
ax[1].legend(methods, loc='upper left')
leg = ax[1].get_legend()
for m in range(M):
    leg.legend_handles[m].set_color(colors[m])
ax[s].set_xticklabels(filenames)
ax[s].tick_params(axis='x', labelrotation=45)
fig.subplots_adjust(hspace=0.3)
#plt.savefig('../figures/audio_snr_all.pdf',bbox_inches='tight')
plt.show()
