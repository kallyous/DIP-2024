import numpy as np
import matplotlib.pyplot as plt
theta = np.linspace(0, 2*np.pi, 100)
r = np.ones_like(theta)
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.set_yticklabels([])
colormap = plt.get_cmap('hsv')
ax.scatter(theta, r, c=theta, cmap=colormap, linewidth=2)
plt.show()