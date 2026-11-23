import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
target_distance = np.linspace(0, 100, 100)
dist_threshold = 23
kernel = (target_distance <= dist_threshold).astype(float) * np.exp(-0.5 * (target_distance / dist_threshold) ** 2)
# kernel = np.exp(-0.5 * (kernel / dist_threshold) ** 2)

fig, ax = plt.subplots()
ax.plot(target_distance, kernel)
# %%
