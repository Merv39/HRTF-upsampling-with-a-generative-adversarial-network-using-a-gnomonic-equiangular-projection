from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import visualisation as vis
from pathlib import Path
import matplotlib.pyplot as plt

# GAN vs HRTF Selection
hrtf_2_loc = Path(r"Z:\home\HRTF-GANs-27Sep22-prep-for-publicationadversarial-network-using-a-gnomonic-equiangular-projection\baseline_results\Sonicom\hrtf_selection\valid\sofa_min_phase\maximum.sofa")
hrtf_1_loc = Path(r"Z:\home\HRTF-GANs-27Sep22-prep-for-publicationadversarial-network-using-a-gnomonic-equiangular-projection\runs-hpc\ari-upscale-4\valid\nodes_replaced\sofa_min_phase\Sonicom_10.sofa")
hrtf_1_loc = Path(r"Z:\home/HRTF-GANs-27Sep22-prep-for-publicationadversarial-network-using-a-gnomonic-equiangular-projection/baseline_results/Sonicom/barycentric/valid/barycentric_interpolated_data_2/nodes_replaced/sofa_min_phase/Sonicom_191.sofa")

# hrtf1, hrtf2 = ld.load_example_sofa_files()
hrtf1 = ld.HRTF(hrtf_1_loc)
hrtf2 = ld.HRTF(hrtf_2_loc)

#HRTFs need to have the same number of points to be matched
print(len(hrtf1.locs))
print(len(hrtf2.locs))

hrtf1, hrtf2 = ld.match_hrtf_locations(hrtf1, hrtf2)

# Calculating HRTF Metrics
spectra, freqs, phase = hf.hrir2hrtf(hrtf1.hrir, hrtf1.fs)
ild = hf.ild_estimator_rms(hrtf1.hrir)
itd_s, itd_samps, maxiacc = hf.itd_estimator_maxiacce(hrtf1.hrir, hrtf1.fs)

# Visualising HRTF Metrics
# vis.plot_itd_overview(hrtf1)
# vis.plot_ild_overview(hrtf1)
# vis.plot_tf_overview(hrtf1, az=[0, 90, 180, 270])
vis.plot_tf_overview(hrtf1, az=[0, 90, 180, 270])
# vis.plot_source_locations(hrtf1.locs)
plt.show()