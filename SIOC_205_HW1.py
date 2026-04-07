from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# File paths
ocean_file = Path("/Users/auroraczajkowski/Desktop/SIOC 205 /HW 1/M2_ocean.txt")
mean_file = Path("/Users/auroraczajkowski/Desktop/SIOC 205 /HW 1/M2_mean.txt")

# Load data
ocean = np.genfromtxt(ocean_file, delimiter=",", names=True)
mean = np.genfromtxt(mean_file, delimiter=",", names=True)

# Create figure with 2 panels
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

# Top panel: M2_ocean
axes[0].plot(ocean["jday"], ocean["depth"], color="steelblue", linewidth=1.5)
axes[0].set_title("M2 Ocean")
axes[0].set_ylabel("Depth (m)")
axes[0].grid(True, alpha=0.3)

# Bottom panel: M2_mean
axes[1].plot(mean["jday"], mean["depth"], color="darkorange", linewidth=1.5)
axes[1].set_title("M2 Mean")
axes[1].set_xlabel("Julian Day")
axes[1].set_ylabel("Depth (m)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
