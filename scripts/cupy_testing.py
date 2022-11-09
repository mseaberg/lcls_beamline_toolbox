#import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append('/cds/home/s/seaberg/dev/lcls_beamline_toolbox')
from lcls_beamline_toolbox.xraybeamline2d import beam, util, optics, beamline2d


N = 4096
E0 = 10000

beam_params = {
        'photonEnergy': E0,
        'N': N,
        'sigma_x': 23e-6,
        'sigma_y': 23e-6,
        'rangeFactor': 5,
        'scaleFactor': 10,
        'z_source': 650-26
        }

b1 = beam.Beam(beam_params=beam_params)

im0 = optics.PPM('im0',z=700)
m1 = optics.Mirror('m1',z=800, length=0.2)

im1 = optics.PPM('im1',z=900)

devices = [im0,m1,im1]

test_beamline = beamline2d.Beamline(devices)

tic = time.perf_counter()
for i in range(10):
    b2 = test_beamline.propagate_beamline(b1)

toc = time.perf_counter()



print(toc-tic)


test_beamline.im1.view_beam()

#output = cp.asnumpy(b2.wave)
output = b2.wave

plt.figure()
plt.imshow(np.abs(output))
plt.figure()
plt.imshow(np.angle(output))
plt.show()
