# from https://python-graph-gallery.com/197-available-color-palettes-with-matplotlib/

# library & dataset
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd


# Data
import seaborn as sns

df = sns.load_dataset('iris')

# We use the specie column to choose the color. We need to make a numerical vector from it:
df['species'] = pd.Categorical(df['species'])
#df['species'].cat.codes

# Scatter
plt.scatter(df['sepal_length'], df['sepal_width'], s=62, c=df['species'].cat.codes, cmap="Set2", alpha=0.9, linewidth=0)

plt.show()
