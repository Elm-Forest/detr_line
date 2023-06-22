# Let's take a look at the segmentation map we got
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open('D:\dataset\power_line_set/train/mask/2_mask.jpg')
plt.figure(figsize=(8, 6))
im = plt.imshow(np.array(img.convert('RGB')))
palette = [[255, 255, 255], [0, 0, 0]]
classes = ('line', 'fg obj')
# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=np.array(palette[i]) / 255.,
                          label=classes[i]) for i in range(2)]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
           fontsize='large')

plt.show()
