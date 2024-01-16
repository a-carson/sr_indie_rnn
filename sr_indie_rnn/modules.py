import torch

class RNN(torch.nn.Module):

    def __init__(self, cell_type, hidden_size, in_channels=1, out_channels=1, residual_connection=True, os_factor=1.0):
        super().__init__()
        if cell_type == 'gru':
            self.rec = torch.nn.GRU(input_size=in_channels, hidden_size=hidden_size, batch_first=True)
        elif cell_type == 'lstm':
            self.rec = torch.nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True)
        elif cell_type == 'euler_gru':
            self.rec = VariableSampleRateGRU(input_size=in_channels, hidden_size=hidden_size, batch_first=True, os_factor=os_factor, improved_euler=False)
        elif cell_type == 'improved_euler_gru':
            self.rec = VariableSampleRateGRU(input_size=in_channels, hidden_size=hidden_size, batch_first=True, os_factor=os_factor, improved_euler=True)
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

class VariableSampleRateGRU(torch.nn.GRU):
    def __init__(self, input_size, hidden_size,
                 bias=True, batch_first=False,
                 os_factor=1.0, improved_euler=False):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         bias=bias,
                         batch_first=batch_first)
        self.improved = improved_euler
        self.os_factor = os_factor
        self.d_line_method = False
        self.cell = torch.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

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

        return states, h.view(1, batch_size, self.hidden_size)

    def nonlinearity(self, x, h):
        h_next = self.cell(x, h)
        return h_next - h

class VariableDelayLineGRU(torch.nn.GRU):
    def __init__(self, input_size, hidden_size,
                 bias=True, batch_first=False,
                 os_factor=1.0, improved_euler=False):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         bias=bias,
                         batch_first=batch_first)
        self.improved = improved_euler
        self.os_factor = os_factor
        self.d_line_method = False
        self.cell = torch.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

    def forward(self, x, h=None):

        batch_size = x.shape[0]
        num_samples = x.shape[1]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h.view(batch_size, self.hidden_size)

        states = torch.zeros(batch_size, num_samples, self.hidden_size, device=x.device)

        if self.improved:
            num_samples -= 1

        for i in range(x.shape[1]):
            h_read = states[:, i - self.os_factor, :]
            xi = x[:, i, :]
            nl = self.nonlinearity(xi, h_read)
            h = h_read + nl
            states[:, i, :] = h

        return states, h.view(1, batch_size, self.hidden_size)

    def nonlinearity(self, x, h):
        h_next = self.cell(x, h)
        return h_next - h


# inference only -- legacy --------
class VariableSampleRateGRUInference:
    def __init__(self, parent: torch.nn.Module):
        self.weight_hh_l0 = parent.rec.weight_hh_l0
        self.weight_ih_l0 = parent.rec.weight_ih_l0
        self.bias_hh_l0 = parent.rec.bias_hh_l0
        self.bias_ih_l0 = parent.rec.bias_ih_l0
        self.linear = parent.linear
        self.hidden_size = parent.rec.hidden_size
        self.improved = False

    def forward(self, u, k=1.0, d_line=False):
        with torch.no_grad():
            # Init State
            x = u.clone()
            states = torch.zeros(x.shape[1], self.hidden_size)
            if d_line:
                tau = int(1/k)
                for i in range(x.shape[1]):
                    h_read = states[i-tau, :]
                    xi = x[:, i, 0]
                    nl = self.nonlinearity(h_read, xi)
                    h = h_read + nl
                    states[i, :] = h

            else:
                h = torch.zeros(self.hidden_size)
                num_samples = x.shape[1]
                if self.improved:
                    num_samples -= 1

                for i in range(num_samples):
                    xi = x[:, i, 0]

                    if self.improved:
                        xi_next = x[:, i+1, 0]
                        f = self.nonlinearity(h, xi)
                        h_guess = h + k * f
                        h += k/2 * (f + self.nonlinearity(h_guess, xi_next))
                    else:
                        nl = self.nonlinearity(h, xi)
                        h += k * nl

                    states[i, :] = h

            return self.linear(states) + x, 0

    def nonlinearity(self, h, x):
        Wh = self.weight_hh_l0 @ h + self.bias_hh_l0
        Wi = self.weight_ih_l0 @ x + self.bias_ih_l0
        r = torch.sigmoid(Wi[:self.hidden_size] + Wh[:self.hidden_size])
        z = torch.sigmoid(
            Wi[self.hidden_size:2 * self.hidden_size] + Wh[self.hidden_size:2 * self.hidden_size])
        n = torch.tanh(Wi[self.hidden_size * 2:self.hidden_size * 3] + r * Wh[
                                                                           self.hidden_size * 2:self.hidden_size * 3])
        return (1 - z) * (n - h)


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





