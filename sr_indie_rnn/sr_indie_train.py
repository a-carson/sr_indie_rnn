import torchaudio.transforms
import pytorch_lightning as pl
import time
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import utils.loss_modules
import sr_indie_rnn.modules as srirnn
import itertools


class BaselineRNN(pl.LightningModule):
    def __init__(self,
                 rnn_model: torch.nn.Module,
                 loss_module: torch.nn.Module,
                 sample_rate: int,
                 tbptt_steps: int = 1024,
                 learning_rate: float = 5e-4,
                 use_wandb: bool = False,
                 log_audio_every_n_epochs: int = 10,):

        super().__init__()
        self.model = rnn_model
        self.sample_rate = sample_rate
        self.truncated_bptt_steps = tbptt_steps
        self.save_hyperparameters()
        self.loss_module = loss_module
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb
        self.last_time = time.time()
        self.log_audio_every_n_epochs = log_audio_every_n_epochs

        self.automatic_optimization = False
        self.sanity_check_val = True
        self.transient_trim = sample_rate


    def forward_frames(self, x, frame_size=None):
        y_pred = torch.zeros_like(x)
        if frame_size is None:
            frame_size = self.sample_rate
        num_frames = int(np.floor(x.shape[1] / frame_size))

        # process in 1s frames
        for n in range(num_frames):
            start = frame_size * n
            end = frame_size * (n+1)
            x_frame = x[:, start:end, :]
            y_pred_frame, _ = self.model(x_frame)
            y_pred[:, start:end, :] = y_pred_frame

        return y_pred

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        x, y = batch
        num_frames = int(np.floor(x.shape[1] / self.truncated_bptt_steps))

        for n in range(num_frames):
            opt.zero_grad()
            warmup_step = n == 0

            start = self.truncated_bptt_steps * n
            end = self.truncated_bptt_steps * (n+1)
            x_frame = x[:, start:end, :]
            y_frame = y[:, start:end, :]

            if warmup_step:
                self.model.reset_state()
                self.last_time = time.time()
            else:
                self.model.detach_state()

            y_pred, last_state = self.model(x_frame)

            if not warmup_step:
                loss = self.loss_module(y_frame, y_pred, high_pass=True)
                self.manual_backward(loss)
                opt.step()
                self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.model.reset_state()
        y_pred = self.forward_frames(x)
        loss = self.loss_module(y[:, self.transient_trim:, :],
                                y_pred[:, self.transient_trim:, :])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # SAVE AUDIO
        if (self.current_epoch % self.log_audio_every_n_epochs) == 0:
            self.log_audio('Val_A', y_pred[0, :, :])
            self.log_audio('Val_B', y_pred[int(x.shape[0]/2), :, :])
            self.log_audio('Val_C', y_pred[-1, :, :])

        if self.sanity_check_val:
            if self.use_wandb:
                wandb.log({'val_loss_og': loss.detach().cpu().numpy()})
            self.log_audio('Val_A_target', y[0, :, :])
            self.log_audio('Val_B_target', y[int(x.shape[0] / 2), :, :])
            self.log_audio('Val_C_target', y[-1, :, :])
            self.sanity_check_val = False


    def test_step(self, batch, batch_idx):
        x, y = batch
        self.model.reset_state()
        y_pred = self.forward_frames(x)
        loss = self.loss_module(y[:,      self.transient_trim:, :],
                                y_pred[:, self.transient_trim:, :])
        self.log("test_loss", loss, on_epoch=True, prog_bar=False, logger=True)

        # SAVE AUDIO
        self.log_audio('Test', torch.flatten(y_pred))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-08, weight_decay=0)


    def log_audio(self, caption, audio):
        if self.use_wandb:
            wandb.log({'Audio/' + caption: wandb.Audio(audio.cpu().detach(), caption=caption, sample_rate=self.sample_rate),
                      'epoch': self.current_epoch})

class PretrainedRNN(BaselineRNN):
    def __init__(self,
                 rnn_model_json: str,
                 loss_module: torch.nn.Module,
                 sample_rate: int,
                 tbptt_steps: int = 1024,
                 learning_rate: float = 5e-4,
                 use_wandb: bool = False,
                 log_audio_every_n_epochs: int = 10):
        super().__init__(rnn_model=srirnn.get_AudioRNN_from_json(rnn_model_json),
                         loss_module=loss_module,
                         sample_rate=sample_rate,
                         tbptt_steps=tbptt_steps,
                         learning_rate=learning_rate,
                         use_wandb=use_wandb,
                         log_audio_every_n_epochs=log_audio_every_n_epochs)

class FIRInterpRNN(PretrainedRNN):
    def __init__(self,
                 order: int,
                 rnn_model_json: str,
                 loss_module: torch.nn.Module,
                 sample_rate: int,
                 double_precision: bool = False,
                 base_sample_rate = None,
                 tbptt_steps: int = 1024,
                 learning_rate: float = 5e-4,
                 use_wandb: bool = False,
                 log_audio_every_n_epochs: int = 10):
        super().__init__(rnn_model_json, loss_module, sample_rate, tbptt_steps, learning_rate, use_wandb, log_audio_every_n_epochs)
        cell = srirnn.get_cell_from_rnn(self.model.rec)
        if base_sample_rate is not None:
            os_factor = np.double(sample_rate) / np.double(base_sample_rate)
        else:
            os_factor = 1
        self.model.rec = srirnn.LagrangeInterp_RNN(cell=cell, order=order, os_factor=os_factor)
        for p in itertools.chain(self.model.rec.cell.parameters(), self.model.linear.parameters()):
            p.requires_grad = False

        if double_precision:
            self.model.double()
        else:
            self.model.float()
        print(self.model)


class SRIndieRNN(pl.LightningModule):
    def __init__(self,
                 rnn_model: torch.nn.Module,
                 loss_module: torch.nn.Module,
                 sample_rate: int,
                 tbptt_steps: int = 1024,
                 learning_rate: float = 5e-4,
                 use_wandb: bool = False):

        super().__init__()
        self.model = rnn_model
        self.sample_rate = sample_rate
        self.truncated_bptt_steps = tbptt_steps
        self.save_hyperparameters()
        self.loss_module = loss_module
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb
        self.last_time = time.time()

        self.automatic_optimization = False


    def forward(self, x, factor):

        return self.model(
            torch.cat((x, factor.repeat(x.shape)), dim=-1)
        )

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        x, y = batch

        factor = 1 + torch.randint(low=0, high=24, size=(1, 1), device=x.device) / 8
        ratio = float(factor).as_integer_ratio()
        x = self.resampler(x, ratio)
        y = self.resampler(y, ratio)
        tbptt_steps = self.truncated_bptt_steps * ratio[0] // ratio[1]

        num_frames = int(np.floor(x.shape[1] / tbptt_steps))

        for n in range(num_frames):
            opt.zero_grad()
            warmup_step = n == 0

            start = tbptt_steps * n
            end = tbptt_steps * (n+1)
            x_frame = x[:, start:end, :]
            y_frame = y[:, start:end, :]

            if warmup_step:
                self.model.reset_state()
                self.last_time = time.time()
            else:
                self.model.detach_state()

            y_pred, last_state = self.forward(x_frame, factor)

            if not warmup_step:
                loss = self.loss_module(y_frame, y_pred, high_pass=True)
                self.manual_backward(loss)
                opt.step()
                self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.model.reset_state()

        y_pred_44k, loss = self.validate_at_sample_rate(x, y, torch.ones(1, device=x.device))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        y_pred_88k, loss = self.validate_at_sample_rate(x, y, 2 * torch.ones(1, device=x.device))
        self.log("val_loss_88k", loss, on_epoch=True, prog_bar=True, logger=True)

        # SAVE AUDIO (at 44k)
        if self.current_epoch == 0:
            self.log_audio('Val_A_target', y[0, :, :], self.sample_rate)
            self.log_audio('Val_B_target', y[int(x.shape[0] / 2), :, :], self.sample_rate)
            self.log_audio('Val_C_target', y[-1, :, :], self.sample_rate)

        self.log_audio('Val_A_88k', y_pred_88k[0, :, :], self.sample_rate * 2)
        self.log_audio('Val_B_88k', y_pred_88k[int(x.shape[0]/2), :, :], self.sample_rate * 2)
        self.log_audio('Val_C_88k', y_pred_88k[-1, :, :], self.sample_rate * 2)


    def validate_at_sample_rate(self, x, y, factor):
        ratio = float(factor).as_integer_ratio()
        x = self.resampler(x, ratio)
        y = self.resampler(y, ratio)
        y_pred, _ = self.forward(x, factor)
        loss = self.loss_module(y, y_pred)
        return y_pred, loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.model.reset_state()
        y_pred, _ = self.forward(x, torch.ones(1))
        loss = self.loss_module(y, y_pred)
        self.log("test_loss", loss, on_epoch=True, prog_bar=False, logger=True)

        # SAVE AUDIO
        self.log_audio('Test', torch.flatten(y_pred), self.sample_rate)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-08, weight_decay=0)


    def log_audio(self, caption, audio, sample_rate):
        if self.use_wandb:
            wandb.log({'Audio/' + caption: wandb.Audio(audio.cpu().detach(), caption=caption, sample_rate=sample_rate),
                      'epoch': self.current_epoch})

    def resampler(self, x, ratio: tuple):
        return torchaudio.functional.resample(x.permute(0, 2, 1),
                                              orig_freq=ratio[1],
                                              new_freq=ratio[0]).permute(0, 2, 1)

class RNNtoSTN(pl.LightningModule):
    def __init__(self,
                 ckpt_path,
                 rnn_model: torch.nn.Module,
                 loss_module: torch.nn.Module,
                 sample_rate: int,
                 tbptt_steps: int = 1024,
                 learning_rate: float = 5e-4,
                 use_wandb: bool = False):

        super().__init__()
        # child model
        child = rnn_model
        # parent model
        pretrained_pl_model = BaselineRNN.load_from_checkpoint(ckpt_path, map_location=self.device)
        self.model = pretrained_pl_model.model

        # copy weights to variable SR model
        child.rec.cell.weight_hh = self.model.rec.weight_hh_l0
        child.rec.cell.weight_ih = self.model.rec.weight_ih_l0
        child.rec.cell.bias_hh = self.model.rec.bias_hh_l0
        child.rec.cell.bias_ih = self.model.rec.bias_ih_l0
        self.model.rec = child.rec

        self.sample_rate = sample_rate
        self.model.rec.os_factor = sample_rate / pretrained_pl_model.sample_rate
        self.truncated_bptt_steps = tbptt_steps
        self.save_hyperparameters()
        self.loss_module = loss_module
        self.mrsl = utils.loss_modules.MRSL(sample_rate=sample_rate / int(self.model.rec.os_factor))        # rounded
        self.spec_loss = utils.loss_modules.SpectralLoss(n_fft=2048, win_length=2048, hop_length=512)
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb
        self.last_time = time.time()

        self.automatic_optimization = False



    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        x, y = batch
        num_frames = int(np.floor(x.shape[1] / self.truncated_bptt_steps))

        for n in range(num_frames):
            opt.zero_grad()
            warmup_step = n == 0

            start = self.truncated_bptt_steps * n
            end = self.truncated_bptt_steps * (n+1)
            x_frame = x[:, start:end, :]
            y_frame = y[:, start:end, :]

            if warmup_step:
                self.model.reset_state()
                self.last_time = time.time()
            else:
                self.model.detach_state()

            y_pred, last_state = self.model(x_frame)

            if not warmup_step:
                loss = self.loss_module(y_frame, y_pred, high_pass=True)
                self.manual_backward(loss)
                opt.step()
                self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = torch.zeros_like(y)
        self.model.reset_state()
        frame_size = self.sample_rate
        num_frames = int(np.floor(x.shape[1] / frame_size))

        # process in 1s frames
        for n in range(num_frames):
            start = frame_size * n
            end = frame_size * (n+1)
            x_frame = x[:, start:end, :]
            y_pred_frame, _ = self.model(x_frame)
            y_pred[:, start:end, :] = y_pred_frame
            self.model.detach_state()


        loss = self.loss_module(y, y_pred)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # SAVE AUDIO
        if self.current_epoch == 0:
            self.log_audio('Val_A_target', y[0, :, :])
            self.log_audio('Val_B_target', y[int(x.shape[0] / 2), :, :])
            self.log_audio('Val_C_target', y[-1, :, :])

        self.log_audio('Val_A', y_pred[0, :, :])
        self.log_audio('Val_B', y_pred[int(x.shape[0]/2), :, :])
        self.log_audio('Val_C', y_pred[-1, :, :])


    def test_step(self, batch, batch_idx):
        x, y = batch
        self.model.reset_state()
        y_pred, _ = self.model(x)

        os = int(self.model.rec.os_factor)
        y = torchaudio.functional.resample(y, orig_freq=os, new_freq=1)
        y_pred = torchaudio.functional.resample(y_pred, orig_freq=os, new_freq=1)

        loss = self.loss_module(y, y_pred)
        spec_conv, mag, err = self.spec_loss(y.squeeze(-1), y_pred.squeeze(-1))
        self.log("test_loss_esr", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("spec_conv", spec_conv, on_epoch=True, prog_bar=False, logger=True)
        self.log("spec_mag", mag, on_epoch=True, prog_bar=False, logger=True)
        freqs = self.sample_rate / os / self.spec_loss.n_fft * torch.arange(0, self.spec_loss.n_fft // 2 + 1)
        plt.plot(freqs, 20 * torch.log10(err).cpu())
        plt.ylim([-60, 0])
        plt.xlabel('Freq [Hz]')
        plt.ylabel('dB')
        if self.use_wandb:
            wandb.log({"Spectral error": plt})
        else:
            plt.show()
        # SAVE AUDIO
        self.log_audio('Test_target', torch.flatten(y))
        self.log_audio('Test', torch.flatten(y_pred))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-08, weight_decay=0)


    def log_audio(self, caption, audio):
        if self.use_wandb:
            wandb.log({'Audio/' + caption: wandb.Audio(audio.cpu().detach(), caption=caption, sample_rate=int(self.sample_rate * self.model.rec.os_factor)),
                      'epoch': self.current_epoch})

class BaselineCNN(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 loss_module: torch.nn.Module,
                 sample_rate: int,
                 learning_rate: float = 5e-4,
                 use_wandb: bool = False):

        super().__init__()
        self.model = model
        self.sample_rate = sample_rate
        self.save_hyperparameters()
        self.loss_module = loss_module
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb
        self.last_time = time.time()
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        x, y = batch

        opt.zero_grad()

        y_pred = self.forward(x)
        loss = self.loss_module(y, y_pred, high_pass=True)
        self.manual_backward(loss)
        opt.step()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_module(y, y_pred)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # SAVE AUDIO
        if self.current_epoch == 0:
            self.log_audio('Val_A_target', y[0, :, :])
            self.log_audio('Val_B_target', y[int(x.shape[0] / 2), :, :])
            self.log_audio('Val_C_target', y[-1, :, :])

        self.log_audio('Val_A', y_pred[0, :, :])
        self.log_audio('Val_B', y_pred[int(x.shape[0]/2), :, :])
        self.log_audio('Val_C', y_pred[-1, :, :])


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_module(y, y_pred)
        self.log("test_loss", loss, on_epoch=True, prog_bar=False, logger=True)

        # SAVE AUDIO
        self.log_audio('Test', torch.flatten(y_pred))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-08, weight_decay=0)


    def log_audio(self, caption, audio):
        if self.use_wandb:
            wandb.log({'Audio/' + caption: wandb.Audio(audio.cpu().detach(), caption=caption, sample_rate=self.sample_rate),
                      'epoch': self.current_epoch})


# class IndieRNN(pl.LightningModule):
#     def __init__(self,
#                  rnn_model: torch.nn.Module,
#                  loss_module: torch.nn.Module,
#                  sample_rate: int,
#                  tbptt_steps: int = 1024,
#                  learning_rate: float = 5e-4):
#
#         super().__init__()
#         self.model = rnn_model
#         self.sample_rate = sample_rate
#         self.truncated_bptt_steps = tbptt_steps
#         self.save_hyperparameters()
#         self.loss_module = loss_module
#         self.os_factor = 1.0
#         self.learning_rate = learning_rate
#
#
#     def training_step(self, batch, batch_idx):
#         warmup_step = [batch_idx == 0]
#         x, y = batch
#
#         if self.os_factor > 1.0:
#             x = self.resampler(x)
#             y = self.resampler(y)
#         y_pred, states = self(x)
#         hiddens = states[:, -1, :]
#
#         if warmup_step:
#             loss = torch.zeros(1, device=self.device, requires_grad=True)
#         else:
#             loss = self.loss_function(y, y_pred, high_pass=True)
#             self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#
#         new_time = time.time()
#         self.log("time_per", new_time - self.last_time, on_epoch=True, on_step=False, prog_bar=True)
#         return {"loss": loss, "hiddens": hiddens}
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         self.model.reset_state()
#         y_pred, _ = self.model(x)
#         loss = self.loss_module(y, y_pred)
#         self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#
#         x_up = self.resampler(x)
#         y_up = self.resampler(y)
#         y_pred_up, _ = self.model(x_up, hiddens=None)
#         loss_up = self.loss_module(y_up, y_pred_up)
#         self.log("val_loss_OS", loss_up, on_epoch=True, prog_bar=True, logger=True)
#
#         # SAVE AUDIO
#         if self.current_epoch % 250 == 0:
#             self.log_audio('Val_A', y_pred[0, :, :])
#             self.log_audio('Val_B', y_pred[int(x.shape[0]/2), :, :])
#             self.log_audio('Val_C', y_pred[-1, :, :])
#
#
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred, _ = self.forward(x, hiddens=None)
#         loss = self.loss_function(y, y_pred)
#         self.log("test_loss", loss, on_epoch=True, prog_bar=False, logger=True)
#
#         # SAVE AUDIO
#         audio_out = torch.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1], 1))
#         self.log_audio('Test', audio_out)
#
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-08, weight_decay=0)
#
#     def on_train_epoch_start(self):
#         self.last_time = time.time()
#         os_factor = np.random.randint(0, 4) + 1
#         # self.resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate,
#         #                                                 new_freq=os_factor * self.sample_rate)
#         self.model.rec.time_step = 1/(self.sample_rate * os_factor)
#         self.log("os", os_factor/1.0, on_epoch=True, prog_bar=True)
#         print('os=', os_factor)
#         self.os_factor = os_factor
#
#     def resampler(self, x):
#         return torchaudio.functional.resample(x.permute(0, 2, 1),
#                                               orig_freq=self.sample_rate,
#                                               new_freq=self.os_factor * self.sample_rate).permute(0, 2, 1)
#
#
#
#
#
#     def log_audio(self, caption, audio):
#         if self.use_wandb:
#             wandb.log({'Audio/' + caption: wandb.Audio(audio.cpu().detach(), caption=caption, sample_rate=self.sample_rate),
#                       'epoch': self.current_epoch})




