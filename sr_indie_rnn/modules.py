import torch
import numpy as np

class RNN(torch.nn.Module):

    def __init__(self, cell_type, hidden_size, in_channels=1, out_channels=1, residual_connection=True, os_factor=1.0):
        super().__init__()
        if cell_type == 'gru':
            self.rec = torch.nn.GRU(input_size=in_channels, hidden_size=hidden_size, batch_first=True)
        elif cell_type == 'lstm':
            self.rec = torch.nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True)
        elif cell_type == 'rnn':
            self.rec = torch.nn.RNN(hidden_size=hidden_size, input_size=in_channels, batch_first=True)
        else:
            # variable sample rate GRU types
            cell = torch.nn.GRUCell(input_size=in_channels, hidden_size=hidden_size, bias=True)
            if cell_type == 'euler_gru':
                self.rec = STNRNN(cell=cell, improved_euler=False, os_factor=os_factor)
            elif cell_type == 'improved_euler_gru':
                cell = torch.nn.GRUCell(input_size=in_channels, hidden_size=hidden_size, bias=True)
                self.rec = STNRNN(cell=cell, os_factor=os_factor, improved_euler=True)
            elif cell_type == 'delay_line':
                self.rec = DelayLineRNN(cell=cell, os_factor=os_factor)

        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=out_channels)
        self.residual = residual_connection
        self.state = None

    def forward(self, x):

        states, last_state = self.rec(x, self.state)
        out = self.linear(states)
        if self.residual:
            out += x[..., 0].unsqueeze(-1)

        self.state = last_state
        return out, states

    def reset_state(self):
        self.state = None

    def detach_state(self):
        self.state = self.state.detach()


class RNNCell(torch.nn.Module):
    def __init__(self, cell_type, hidden_size, in_size):
        super().__init__()
        if cell_type == 'gru':
            self.rec = torch.nn.GRUCell(hidden_size=hidden_size, input_size=in_size)
        elif cell_type == 'lstm':
            self.rec = torch.nn.LSTMCell(hidden_size=hidden_size, input_size=in_size)
        else:
            self.rec = torch.nn.RNNCell(hidden_size=hidden_size, input_size=in_size)

    def forward(self, x, h0=None):
        return self.rec(x, h0)

class STNRNN(torch.nn.Module):
    def __init__(self, cell: torch.nn.RNNCellBase,
                 os_factor=1.0,
                 improved_euler=False):
        super().__init__()
        self.os_factor = os_factor
        self.cell = cell
        self.hidden_size = self.cell.hidden_size

    def forward(self, x, h=None):

        batch_size = x.shape[0]
        num_samples = x.shape[1]
        k = 1/self.os_factor

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h.view(batch_size, self.hidden_size)

        states = torch.zeros(batch_size, num_samples, self.hidden_size, device=x.device)

        if self.improved:
            num_samples -= 1

        for i in range(num_samples):

            xi = x[:, i, :]

            if self.improved:
                xi_next = x[:, i + 1, :]
                f = self.nonlinearity(xi, h)
                h_next_guess = h + k * f
                h = h + k / 2 * (f + self.nonlinearity(xi_next, h_next_guess))
            else:
                nl = self.nonlinearity(xi, h)
                h = h + k * nl

            states[:, i, :] = h

        if self.improved:
            states = torch.roll(states, int(1 - self.os_factor), dims=1)

        return states, h.view(1, batch_size, self.hidden_size)

    def nonlinearity(self, x, h):
        h_next = self.cell(x, h)
        return h_next - h

class DelayLineRNN(torch.nn.Module):
    def __init__(self, cell: torch.nn.RNNCellBase,
                 os_factor=1.0):
        super().__init__()
        self.os_factor = os_factor
        self.cell = cell
        self.hidden_size = self.cell.hidden_size

    def forward(self, x, h=None):

        batch_size = x.shape[0]
        num_samples = x.shape[1]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h.view(batch_size, self.hidden_size)

        states = torch.zeros(batch_size, num_samples, self.hidden_size, device=x.device)

        # lin. interp setup
        delay_near = int(np.floor(self.os_factor))
        delay_far = int(np.ceil(self.os_factor))
        alpha = self.os_factor - delay_near

        for i in range(x.shape[1]):
            # lin. interp
            h_read = (1 - alpha) * states[:, i - delay_near, :] + alpha * states[:, i - delay_far, :]

            xi = x[:, i, :]
            h = self.cell(xi, h_read)
            states[:, i, :] = h

        return states, h.view(1, batch_size, self.hidden_size)

class HybridSTNDelayLineRNN(torch.nn.Module):
    def __init__(self, cell: torch.nn.RNNCellBase,
                        os_factor=1.0):
        super().__init__()
        self.os_factor = os_factor
        self.cell = cell
        self.hidden_size = self.cell.hidden_size

    def forward(self, x, h=None):

        batch_size = x.shape[0]
        num_samples = x.shape[1]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h.view(batch_size, self.hidden_size)

        states = torch.zeros(batch_size, num_samples, self.hidden_size, device=x.device)

        # linterp
        delay = int(np.floor(self.os_factor))
        fractional_os_factor = self.os_factor / delay
        k = 1/fractional_os_factor

        for i in range(x.shape[1]):
            h_read = states[:, i - delay, :]
            xi = x[:, i, :]
            nl = self.nonlinearity(xi, h_read)
            h = h_read + k * nl
            states[:, i, :] = h

        return states, h.view(1, batch_size, self.hidden_size)

    def nonlinearity(self, x, h):
        h_next = self.cell(x, h)
        return h_next - h


class MLP(torch.nn.Module):
    def __init__(self, in_features=1,
                 out_features=1,
                 width=8,
                 n_hidden_layers=1,
                 activation='tanh',
                 bias=True,
                 out_activation=False):
        super(MLP, self).__init__()

        self.model = torch.nn.Sequential()

        for n in range(n_hidden_layers):
            self.model.append(torch.nn.Linear(in_features=in_features, out_features=width, bias=bias))
            if activation == 'tanh':
                self.model.append(torch.nn.Tanh())
            else:
                self.model.append(torch.nn.ReLU())
            in_features = width

        self.model.append(torch.nn.Linear(in_features=width, out_features=out_features, bias=bias))
        if out_activation:
            self.model.append(torch.nn.Tanh())

    def forward(self, x):
        return self.model(x)


class GatedConv1D(torch.nn.Module):

    """
    Single 1D convolutional layer with residual connection
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 kernel_size,
                 fully_connected):
        super(GatedConv1D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fully_connected = fully_connected

        padding = int((kernel_size - 1) * dilation / 2)

        self.conv = torch.nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels * 2,
                                    kernel_size=kernel_size,
                                    dilation=dilation,
                                    padding=padding)

        self.residual_mix = torch.nn.Conv1d(in_channels=in_channels if fully_connected else out_channels,
                                            out_channels=out_channels if fully_connected else in_channels,
                                            # change to out_channels for more overall params
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)

    def forward(self, x):
        y = self.conv(x)  # (N, 2*C, L)
        # gated activation
        z = torch.tanh(y[:, :self.out_channels, :]) * torch.sigmoid(y[:, self.out_channels:, :])  # (N, C, L)
        if self.fully_connected:
            x = self.residual_mix(x) + z  # (N, 1, L)
        else:
            x = self.residual_mix(z) + x
        return x, z


class GCNBlock(torch.nn.Module):
    """
    Block of serially connected dilated convolutional layers
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers,
                 dilation_growth,
                 fully_connected):
        super(GCNBlock, self).__init__()

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()

        dilations = [dilation_growth ** l for l in range(num_layers)]
        for dilation in dilations:
            self.layers.append(GatedConv1D(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           dilation=dilation,
                                           fully_connected=fully_connected))
            if fully_connected:
                in_channels = out_channels

    # expect shape: (N, C, L)
    def forward(self, x):
        batch_size = x.shape[0]
        length = x.shape[2]

        z = torch.empty([batch_size, self.num_layers * self.out_channels, length]).to(x.device)

        for l, layer in enumerate(self.layers):
            x, zn = layer(x)
            z[:, l * self.out_channels: (l + 1) * self.out_channels, :] = zn

        return x, z


class GCNNet(torch.nn.Module):
    """
    End-to-end GCN network (WaveNet), including linear mixing of all intermediate activations
    """
    def __init__(self,
                 num_blocks=1,
                 layers_per_block=10,
                 in_channels=1,
                 out_channels=1,
                 hidden_channels=16,
                 kernel_size=3,
                 dilation_growth=2,
                 fully_connected=False):
        super(GCNNet, self).__init__()

        self.layers_per_block = layers_per_block
        self.hidden_channels = hidden_channels
        self.blocks = torch.nn.ModuleList()

        for b in range(num_blocks):
            self.blocks.append(GCNBlock(in_channels=in_channels,
                                        out_channels=hidden_channels,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth,
                                        num_layers=layers_per_block,
                                        fully_connected=fully_connected))

        # 1x1 convolution as linear mixer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=hidden_channels * layers_per_block * num_blocks,
                            out_channels=out_channels,
                            kernel_size=1))

    # expect shape (N, C, L)
    def forward(self, x):
        batch_size = x.shape[0]
        length = x.shape[2]

        # init activations tensor with correct size for output mixer
        z = torch.empty([batch_size, self.blocks[-1].in_channels, length]).to(x.device)

        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x)
            z[:,
            n * self.hidden_channels * self.layers_per_block:(n + 1) * self.hidden_channels * self.layers_per_block,
            :] = zn

        # return mixed output
        return self.blocks[-1](z)


class ConvInputRNN(RNN):
    """
    RNN with a CNN appended to the front -- out_channels of the CNN must equal out_channels of RNN
    """
    def __init__(self, conv_module: torch.nn.Module,
        cell_type, hidden_size, in_channels=1, out_channels=1, residual_connection=True, os_factor=1.0):
        super().__init__(cell_type, hidden_size, in_channels, out_channels, residual_connection, os_factor)
        self.conv = conv_module


    # expect input: (N, L, C), same as base RNN class
    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return super().forward(x)







