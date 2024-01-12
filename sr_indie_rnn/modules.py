import torch

class RNN(torch.nn.Module):

    def __init__(self, cell_type, hidden_size, in_channels=1, out_channels=1, residual_connection=True, time_step=1.0):
        super().__init__()
        if cell_type == 'gru':
            self.rec = torch.nn.GRU(input_size=in_channels, hidden_size=hidden_size, batch_first=True)
        elif cell_type == 'lstm':
            self.rec = torch.nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True)
        elif cell_type == 'my_gru':
            self.rec = MyGRU(input_size=in_channels, hidden_size=hidden_size, time_step=time_step)
        else:
            self.rec = torch.nn.RNN(hidden_size=hidden_size, input_size=in_channels, batch_first=True)
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


class MyGRU(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 time_step,
                 input_size=1):
        super().__init__()
        self.gru = torch.nn.GRUCell(input_size=input_size,
                                    hidden_size=hidden_size)
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.A = torch.nn.Parameter(torch.Tensor([0.001]))

    def forward(self, x, state):
        batch_size = x.shape[0]
        length = x.shape[1]

        if state is None:
            state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        residual = state.clone()
        out = torch.empty(batch_size, length, self.hidden_size, device=x.device)

        for i in range(length):
            # residual = self.gru(x[:, i, :], state.clone()) - state
            # state += self.time_step * self.A * residual
            # out[:, i, :] = state
            # #
            # last run
            residual = self.gru(x[:, i, :], residual)
            state += self.A * residual
            out[:, i, :] = state

        return out, state



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





