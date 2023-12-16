from sklearn import mixture
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("./images/cat.jpg").copy()
image = np.asarray(image)
H, W, _ = image.shape
image = np.reshape(image, (-1,3))
print(image.shape)

K = 5
gmm = mixture.GaussianMixture(n_components=K, covariance_type='full', init_params="k-means++")
gmm.fit(image)
labels = gmm.predict(image)

colors = gmm.means_.astype(np.uint8)

N = labels.shape[0]
colors = np.tile(colors[:K][None,...], (N,1,1)) # (N, K, 3)
compare = np.tile(np.arange(0, K, 1)[None,...], (N,1)) # (N, K)
association = np.tile(labels[...,None], (1,K)) # (N, K)
association_mask = np.where(compare==association, 1, 0) # (N, K)
segmentation = colors[association_mask==1,...] # (N, 3)
segmentation = np.reshape(segmentation, (H,W,3))
plt.imshow(segmentation)
plt.show()