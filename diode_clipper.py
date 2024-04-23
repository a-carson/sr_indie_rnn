import torch
import copy


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
    def __init__(self, sample_rate):
        super().__init__()
        self.rec = DiodeClipperRec(sample_rate=sample_rate)

    def forward(self, x, h=None):
        return self.rec(x, h)


class DiodeClipperRec(torch.nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.cell = DiodeClipperCell(sample_rate=sample_rate)

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


    def forward(self, x_now, h):

        x_now = x_now.clone() * self.omega
        iteraton = 0
        step = torch.ones_like(x_now)
        f = self.nonlinear_func(h)
        p = self.k / 2 * f - self.k * x_now - h.clone()

        while (iteraton < self.max_iter) and (step.abs() > self.tol):
            f = self.nonlinear_func(h)
            g = h + self.k / 2 * f + p
            g_deriv = 1.0 + self.k / 2 * self.nonlinear_deriv(h)
            step = g / g_deriv
            h -= step
            iteraton += 1

        return h.clone()

