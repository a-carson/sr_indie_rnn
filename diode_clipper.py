import torch
import numpy as np
import math

class DiodeClipperFull(torch.nn.Module):
    def __init__(self, sample_rate, r=1e3, c=33e-9, i_s=2.52e-9, v_t=25.83e-3, n_i=1.752,
                 os_factor=1.0):
        super().__init__()
        self.os_factor = os_factor
        self.k = 1/sample_rate
        self.r = r
        self.c = c
        self.i_s = i_s
        self.v_t = v_t

        self.omega = 1.0 / (self.r * self.c)
        self.sinh_const = 2 * self.i_s / self.c

        self.max_iter = 50
        self.tol = 1e-9

    def nonlinear_func(self, arg):
        return self.omega * arg + self.sinh_const * torch.sinh(arg / self.v_t)

    def nonlinear_deriv(self, arg):
        return self.omega + self.sinh_const * torch.cosh(arg / self.v_t) / self.v_t

    def forward(self, x):

        batch_size = x.shape[0]
        num_samples = x.shape[1]

        states = x.new_zeros(batch_size, num_samples)

        f_prev = x.new_zeros(1)
        f = x.new_zeros(1)
        x_prev = x.new_zeros(1)
        h = x.new_zeros(1)
        h_prev = x.new_zeros(1)

        # pre process
        x *= self.omega

        for i in range(x.shape[1]):
            x_now = x[..., i]
            iteraton = 0
            step = x.new_ones(1)
            p = self.k / 2 * f_prev - self.k / 2 * (x_now + x_prev) - h_prev

            while (iteraton < self.max_iter) and (step.abs() > self.tol):
                f = self.nonlinear_func(h)
                g = h + self.k / 2 * f + p
                g_deriv = 1.0 + self.k / 2 * self.nonlinear_deriv(h)
                step = g / g_deriv
                h -= step
                iteraton += 1

            states[:, i] = h.clone()
            x_prev = x_now.clone()
            f_prev = f.clone()
            h_prev = h.clone()

        return states


class DiodeClipper(torch.nn.Module):
    def __init__(self, sample_rate, lut_path=None):
        super().__init__()
        self.rec = DiodeClipperRec(sample_rate=sample_rate, lut_path=lut_path)

    def forward(self, x, h=None):
        return self.rec(x, h)


class DiodeClipperRec(torch.nn.Module):
    def __init__(self, sample_rate, lut_path=None):
        super().__init__()
        if lut_path is None:
            self.cell = DiodeClipperCell(sample_rate=sample_rate)
        else:
            self.cell = DiodeClipperLUT(sample_rate=sample_rate, lut_path=lut_path)

    def forward(self, x, h=None):

        batch_size = x.shape[0]
        num_samples = x.shape[1]

        states = x.new_zeros(batch_size, num_samples, 1)
        if h is None:
            h = x.new_zeros(batch_size, 1)

        for i in range(x.shape[1]):
            h = self.cell(x[:, i, :], h.clone())
            states[:, i, :] = h.clone()
        return states, h


class DiodeClipperCell(torch.nn.RNNCellBase):
    def __init__(self, sample_rate, r=1e3, c=33e-9, i_s=2.52e-9, v_t=25.83e-3):
        super().__init__(hidden_size=1, input_size=1, bias=False, num_chunks=1)
        self.k = 1/sample_rate
        self.r = r
        self.c = c
        self.i_s = i_s
        self.v_t = v_t

        self.omega = 1.0 / (self.r * self.c)
        self.sinh_const = 2 * self.i_s / self.c

        self.max_iter = 50
        self.tol = 1e-9

    def nonlinear_func(self, arg):
        return self.omega * arg + self.sinh_const * torch.sinh(arg / self.v_t)

    def nonlinear_deriv(self, arg):
        return self.omega + self.sinh_const * torch.cosh(arg / self.v_t) / self.v_t

    # (input, guess)
    def _nr_solve(self, p, x):
        iteration = 0
        step = torch.ones_like(p)
        while (iteration < self.max_iter) and (step.abs() > self.tol):
            f = self.nonlinear_func(x)
            g = x + self.k / 2 * f + p
            g_deriv = 1.0 + self.k / 2 * self.nonlinear_deriv(x)
            step = g / g_deriv
            x -= step
            iteration += 1
        return x, f




    def forward(self, x_now, h):

        x_now = x_now.clone() * self.omega
        f = self.nonlinear_func(h)
        p = self.k / 2 * f - self.k * x_now - h.clone()
        h, f = self._nr_solve(p, h)

        return h.clone()

class DiodeClipperLUT(DiodeClipperCell):
    def __init__(self, sample_rate, lut_path, r=1e3, c=33e-9, i_s=2.52e-9, v_t=25.83e-3):
        super().__init__(sample_rate, r, c, i_s, v_t)
        lut = torch.from_numpy(np.load(lut_path))
        self.p_ax = lut[:, 0]
        self.x_ax = lut[:, 1]
        self.length = self.x_ax.shape[0]
        self.delta_p = self.p_ax[1] - self.p_ax[0]
        self.order = 5

        idx = torch.arange(self.order + 1, dtype=torch.double)
        self.n_idx = idx.view(-1, 1).repeat(1, self.order)

        mask = torch.ones(self.order+1, self.order+1, dtype=torch.bool)
        mask[torch.eye(self.order + 1, dtype=torch.bool)] = 0
        self.k_idx = idx.view(1, -1).repeat(self.order + 1, 1)[mask].reshape(self.order+1, self.order)
        #self.k_idx = torch.DoubleTensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


    def forward(self, x_now, h):

        x_now = x_now.clone() * self.omega
        f = self.nonlinear_func(h)
        p = self.k / 2 * f - self.k * x_now - h.clone()

        ind_bet = (p - self.p_ax[0]) / self.delta_p + 1
        nn_left = math.floor(ind_bet)
        delta = (ind_bet - nn_left + (self.order+1)//2 - 1).squeeze()

        kernel = torch.prod((delta - self.k_idx) / (self.n_idx - self.k_idx), dim=1)
        x = self.x_ax[nn_left-(self.order+1)//2:nn_left+(self.order+1)//2] @ kernel
        return x

