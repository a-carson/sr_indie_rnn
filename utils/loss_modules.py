import torch
import torch.nn.functional as F


class TimeDomainLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hpf_coeffs = torch.Tensor([-0.85, 1.0]).view(1, 1, -1).detach()

    def forward(self, targ, pred, high_pass=False, batch_mean=True):
        channels = targ.shape[2]
        weight = self.hpf_coeffs.repeat(1, channels, 1).to(targ.device)
        if high_pass:
            targ_hp = F.conv1d(targ.permute(0, 2, 1), weight, padding=1).permute(0, 2, 1)
            pred_hp = F.conv1d(pred.permute(0, 2, 1), weight, padding=1).permute(0, 2, 1)
            esr = esr_loss(targ_hp, pred_hp, batch_mean)
        else:
            esr = esr_loss(targ, pred, batch_mean)
        edc = edc_loss(targ, pred)
        return esr + edc

class StateMatchingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hpf_coeffs = torch.Tensor([-0.85, 1.0]).view(1, 1, -1).detach()

    # expect shape (N, L, C)
    def forward(self, targ, pred, high_pass=True, squish_target=False):

        with torch.no_grad():
            if squish_target:
                state_energy = torch.mean(torch.sum(targ ** 2, dim=1), dim=0)
                top_dawgs = torch.argsort(state_energy, descending=True)[:pred.shape[2]]
                targ = targ[:, :, top_dawgs]

        targ = targ.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        channels = targ.shape[1]
        weight = self.hpf_coeffs.repeat(channels, 1, 1).to(targ.device)
        if high_pass:
            targ = F.conv1d(targ, weight, padding=1, groups=channels)
            pred = F.conv1d(pred, weight, padding=1, groups=channels)

        # calculate ESR over time dimension
        mse = torch.sum((targ - pred)**2, dim=2)
        sigs = torch.sum(targ ** 2, dim=2) + 1e-9

        return torch.mean(mse / sigs) # average over batches


def esr_loss(target, predicted, batch_mean=True):
    signal_energy = torch.mean(target ** 2)
    if batch_mean:
        mse = torch.mean((target - predicted) ** 2)
    else:
        mse = torch.mean((target - predicted) ** 2, dim=-2)

    return mse / signal_energy

def edc_loss(target, predicted):
    err = torch.mean(target - predicted) ** 2
    sig = torch.mean(target ** 2)
    return torch.div(err, sig + 1e-8)

def mae(x1, x2):
    return torch.mean(torch.abs(x1 - x2))

def mse(x1, x2):
    return torch.mean(torch.square(x1 - x2))


class FFTMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, pred):
        target_fft = torch.abs(torch.fft.fft(target))
        pred_fft = torch.abs(torch.fft.fft(pred))
        return mae(target_fft, pred_fft)



"""
Spectral loss
"""
class SpectralLoss(torch.nn.Module):
    def __init__(self,
                 n_fft: int,
                 win_length: int,
                 hop_length: int):
        super(SpectralLoss, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.epsilon = 1e-8

    # requires shapes (N, L)
    def forward(self, target, predicted):
        stft_target = torch.abs(torch.stft(target,
                                           n_fft=self.n_fft,
                                           win_length=self.win_length,
                                           hop_length=self.hop_length,
                                           return_complex=True))
        stft_pred = torch.abs(torch.stft(predicted,
                                           n_fft=self.n_fft,
                                           win_length=self.win_length,
                                           hop_length=self.hop_length,
                                           return_complex=True))

        convergence_loss = torch.norm(stft_target - stft_pred) / (torch.norm(stft_target) + self.epsilon)
        magnitude_loss = torch.norm(torch.log10(stft_target + self.epsilon) - torch.log10(stft_pred + self.epsilon), p=1) / torch.numel(stft_target)
        return convergence_loss + magnitude_loss


"""
Multi Resolution Spectral Loss module
"""


class MRSL(torch.nn.Module):
    def __init__(self,
                 sample_rate,
                 fft_lengths=None,
                 window_sizes=None,
                 overlap=0.25):
        super(MRSL, self).__init__()

        if fft_lengths is None:
            fft_lengths = [4096, 2048, 1024]
        if window_sizes is None:
            window_sizes = [int(0.5 * n) for n in fft_lengths]
        else:
            window_sizes = [int(sample_rate * t) for t in self.window_sizes]

        hop_sizes = [int(w * overlap) for w in window_sizes]

        assert [len(fft_lengths) == len(window_sizes), 'window_sizes and fft_lengths must be the same length']

        self.spec_losses = torch.nn.ModuleList()
        for (n_fft, n_win, hop) in zip(fft_lengths, window_sizes, hop_sizes):
            self.spec_losses.append(SpectralLoss(n_fft=n_fft, win_length=n_win, hop_length=hop))

    def forward(self, target, predicted):
        loss = torch.tensor(0.0)
        for spec_loss in self.spec_losses:
            loss += spec_loss(target, predicted)

        return loss / len(self.spec_losses)


if __name__ == '__main__':
    loss = StateMatchingLoss()
    x = torch.rand(20, 2048, 16)
    y = torch.rand(20, 2048, 8)
    #x[:, 0, :] = torch.ones(20, 4)
    l = loss(x, y)