import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("Lenna.png")
plt.imshow(img)
plt.show()