from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import visualisation as vis
from pathlib import Path
import matplotlib.pyplot as plt
import sofa
import numpy as np

# # GAN vs HRTF Selection
# hrtf_2_loc = Path(r"Z:\home\HRTF-GANs-27Sep22-prep-for-publicationadversarial-network-using-a-gnomonic-equiangular-projection\baseline_results\Sonicom\hrtf_selection\valid\sofa_min_phase\maximum.sofa")
# hrtf_1_loc = Path(r"Z:\home\HRTF-GANs-27Sep22-prep-for-publicationadversarial-network-using-a-gnomonic-equiangular-projection\runs-hpc\ari-upscale-4\valid\nodes_replaced\sofa_min_phase\Sonicom_10.sofa")
# hrtf_1_loc = Path(r"Z:\home/HRTF-GANs-27Sep22-prep-for-publicationadversarial-network-using-a-gnomonic-equiangular-projection/baseline_results/Sonicom/barycentric/valid/barycentric_interpolated_data_2/nodes_replaced/sofa_min_phase/Sonicom_191.sofa")

# # hrtf1, hrtf2 = ld.load_example_sofa_files()
# hrtf1 = ld.HRTF(hrtf_1_loc)
# hrtf2 = ld.HRTF(hrtf_2_loc)

# #HRTFs need to have the same number of points to be matched
# print(len(hrtf1.locs))
# print(len(hrtf2.locs))

# hrtf1, hrtf2 = ld.match_hrtf_locations(hrtf1, hrtf2)

# # Calculating HRTF Metrics
# spectra, freqs, phase = hf.hrir2hrtf(hrtf1.hrir, hrtf1.fs)
# ild = hf.ild_estimator_rms(hrtf1.hrir)
# itd_s, itd_samps, maxiacc = hf.itd_estimator_maxiacce(hrtf1.hrir, hrtf1.fs)

# # Visualising HRTF Metrics
# # vis.plot_itd_overview(hrtf1)
# # vis.plot_ild_overview(hrtf1)
# # vis.plot_tf_overview(hrtf1, az=[0, 90, 180, 270])
# vis.plot_tf_overview(hrtf1, az=[0, 90, 180, 270])
# # vis.plot_source_locations(hrtf1.locs)
# plt.show()

import sys
sys.path.insert(0, '../../src')
import sofa
print(sofa)

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def plot_coordinates(coords, title):
    x0 = coords
    n0 = coords
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0],
                  n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    return q

HRTF_path = r"Z:\home\HRTF-GANs-27Sep22-prep-for-publicationadversarial-network-using-a-gnomonic-equiangular-projection\baseline_results\Sonicom\hrtf_selection\valid\sofa_min_phase\maximum.sofa"
HRTF = sofa.Database.open(HRTF_path)
HRTF.Metadata.dump()

# plot Source positions
source_positions = HRTF.Source.Position.get_values(system="cartesian")
plot_coordinates(source_positions, 'Source positions');

# plot Data.IR at M=5 for E=0
measurement = 5
emitter = 0
legend = []

t = np.arange(0,HRTF.Dimensions.N)*HRTF.Data.SamplingRate.get_values(indices={"M":measurement})

plt.figure(figsize=(15, 5))
for receiver in np.arange(HRTF.Dimensions.R):
    plt.plot(t, HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))
    legend.append('Receiver {0}'.format(receiver))
plt.title('HRIR at M={0} for emitter {1}'.format(measurement, emitter))
plt.legend(legend)
plt.xlabel('$t$ in s')
plt.ylabel(r'$h(t)$')
plt.grid()
plt.show()

HRTF.close()