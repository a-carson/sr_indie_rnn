import torchaudio.transforms
import pytorch_lightning as pl
import time
import torch
import wandb
import numpy as np

class BaselineRNN(pl.LightningModule):
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
        y_pred, _ = self.model(x)
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




