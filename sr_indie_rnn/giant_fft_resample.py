import torch

def giant_fft_upsample(x, orig_freq, new_freq):

    N = x.shape[-1]
    X = torch.fft.fft(x)
    N_up = new_freq * N // orig_freq
    X_up = torch.zeros((1, N_up), dtype=X.dtype)

    X_up[..., 0:N//2] = X[..., 0:N//2].clone()
    X_up[..., N//2] = 0.5 * X[..., N//2].clone()
    X_up[..., N_up - N//2] = 0.5 * X[..., N//2].clone()
    X_up[..., N_up - N // 2 + 1:] = torch.conj(X_up[..., 1:N//2].clone()).flip(-1)
    x_up = torch.fft.ifft(X_up)
    return torch.real(x_up) * new_freq / orig_freq

def giant_fft_downsample(x, orig_freq, new_freq):

    N = x.shape[-1]
    X = torch.fft.fft(x)
    N_down = new_freq * N // orig_freq
    X_down = torch.zeros((1, N_down), dtype=X.dtype)

    X_down[..., 0:N_down//2] = X[..., 0:N_down//2].clone()
    X_down[..., N_down//2] = 0.0
    X_down[..., N_down // 2 + 1:] = torch.conj(X_down[..., 1:N_down//2].clone()).flip(-1)
    x_down = torch.fft.ifft(X_down)
    return torch.real(x_down) * new_freq / orig_freq

# p = 160
# q = 147
# Fs = 44100
# f0 = 10e3
# Fs_up = Fs * p // q
# # t_ax = np.arange(0, Fs) / Fs
# # t_ax_up = np.arange(0, Fs_up) / Fs_up
# #
# # y = np.sin(2 * np.pi * f0 * t_ax)
# # y_up_target = np.sin(2 * np.pi * f0 * t_ax_up)
# # y_up = giant_fft_upsample(y, p, q)
# #
# # plt.plot(y_up_target)
# # plt.plot(y_up)
# # plt.show()
# # esr = np.sum((y_up - y_up_target) ** 2) / np.sum(y_up_target**2)
#
#
# x, sample_rate = torchaudio.load('../../../audio_datasets/dist_fx_192k/44k/test/ht1-input.wav')
# x = x[..., :-1].type(torch.DoubleTensor)
#
#
# x_reaper, sample_rate = torchaudio.load('../../../audio_datasets/dist_fx_192k/48k/test/ht1-input.wav')
# x_pt = torchaudio.functional.resample(x, orig_freq=q, new_freq=p)
#
# x_g = giant_fft_upsample_torch(x, p, q)
# esr = torch.sum((x_pt - x_g) ** 2) / torch.sum(x_g**2)
#
# fvec = sample_rate * np.arange(0, x_g.shape[-1]) / x_g.shape[-1]
# plt.plot(fvec, 10 * np.log10(np.abs(np.fft.fft(x_pt[0, ...].numpy()))))
# plt.plot(fvec, 10 * np.log10(np.abs(np.fft.fft(x_reaper[0, :-1].numpy()))))
# plt.plot(fvec, 10 * np.log10(np.abs(np.fft.fft(x_g[0, ...].numpy()))))
#
#
#
# print(esr)
