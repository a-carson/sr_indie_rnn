from diode_clipper import DiodeClipper
import torch
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    dc = DiodeClipper(sample_rate=44100)
    x_axis = torch.linspace(-1, 1, 2*8192, dtype=torch.double)
    y_axis = torch.zeros_like(x_axis)
    y = y_axis.new_zeros(1, 1)


    for i, x in enumerate(x_axis):
        y, _ = dc.rec.cell._nr_solve(x, y)
        y_axis[i] = y

    lut = torch.stack((x_axis, y_axis), 1).numpy()
    np.save('diode_clipper_LUT_44.1kHz.npy', lut)
    plt.plot(x_axis, y_axis)
    plt.show()
