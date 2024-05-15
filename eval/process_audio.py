import os
import torch
import torchaudio
from utils.giant_fft_resample import giant_fft_resample
import sr_indie_rnn.modules as rnn


audio_path = '../../../audio_datasets/dist_fx_192k/48k/test/input.wav'
model_name = '6505Plus_Red_DirectOut'
base_path = '../Proteus_Tone_Packs/'
os_ratio = [147, 160]       # [new_freq, orig_freq] e.g [160, 147] for SR conversion from 44.1kHz to 48kHz
method = 'lagrange'             # sample rate conversion method
num_samples = 441000            # option to truncate input audio

# load audio
in_sig, base_sample_rate = torchaudio.load(audio_path)
in_sig = in_sig[..., :num_samples]

os_factor = os_ratio[0] / os_ratio[1]

# load models
base_model = rnn.get_AudioRNN_from_json('{}.json'.format(os.path.join(base_path, model_name)))
sr_indie_model = rnn.get_SRIndieRNN(base_model=base_model, method=method)
sr_indie_model.rec.os_factor = os_factor

with torch.no_grad():

    # oversample
    in_sig_os = giant_fft_resample(in_sig, orig_freq=os_ratio[1], new_freq=os_ratio[0])
    #  process
    out_sig_os, _ = sr_indie_model(in_sig_os)
    # downsample
    out_sig = giant_fft_resample(out_sig_os, orig_freq=os_ratio[0], new_freq=os_ratio[1])

    # target
    out_sig_target, _ = base_model(in_sig)

# compute SNR compared to target
torchaudio.save('../audio_out/{}_{}_{}.wav'.format(model_name, os_factor, method), out_sig, base_sample_rate)
torchaudio.save('../audio_out/{}_target.wav'.format(model_name), out_sig_target, base_sample_rate)
diff = out_sig[..., :out_sig_target.shape[-1]].flatten() - out_sig_target.flatten()
snr = out_sig_target.flatten().square().sum() / diff.square().sum()
snr_dB = 10 * torch.log10(snr)
print('SNR = {} dB'.format(snr_dB))

