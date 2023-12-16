from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

seed = int(time.time() / 1e3)
np.random.seed(seed)

BLACK_BOARDER_THICK = 10
WHITE_BOARDER_THICK = 10

image_file = "./images/cat.jpg"

im = Image.open(image_file)
im = np.asarray(im).copy()
H, W, _ = im.shape
im = cv2.resize(im, (int(W/3.45), int(H/3.45)))

def gaussian(x:np.ndarray, mu:np.ndarray, cov:np.ndarray):
    # x: (N, 3)
    # mu: (K, 3)
    # cov: (K, 3, 3)
    # return: (N, K), the pdf value of k Gaussians for each pixel
    
    N = x.shape[0]
    K = mu.shape[0]
    
    x_expanded = np.tile(x[:,None,:], (1,K,1)) # (N, K, 3)
    mu_expanded = np.tile(mu[None,...], (N,1,1)) # (N, K, 3)
    cov_expanded = np.tile(cov[None,...], (N,1,1,1)) # (N, K, 3, 3)
    
    scale_factor = 1.0 / np.sqrt((2*np.pi)**3 * np.linalg.det(cov_expanded)) # (N, K, 3, 3)
    diff = (x_expanded - mu_expanded)[...,None] # (N, K, 3, 1)
    exponent = -0.5 * (np.transpose(diff, (0,1,3,2)) @ np.linalg.inv(cov_expanded) @ diff) # (N, K, 1, 1)
    exponent = np.squeeze(exponent, axis=(2,3)) # (N, K)
    prob = scale_factor * np.exp(exponent) # (N, K)
    
    return prob

def cluster_init(x:np.ndarray, N, K):
    D = 5 / 255.0
    mu = np.zeros((K,3))
    mu[0] = x[np.random.randint(0,N,1)] # pick a random pixel as the first Guassian's mean
   
    for i in range(1,K):
        m = i # m = number of means chosen so far
        
        # -- find a pixel that is most different to all previously chosen means --
        x_expanded = np.tile(x[None,...], (m,1,1)) # (m, N, 3)
        mu_expanded = np.tile(mu[:m][:,None,:], (1,N,1)) # (m, N, 3)
        diff = np.abs(x_expanded - mu_expanded) # (m, N, 3)
        diff_sum = np.sum(diff, axis=(0,2)) # (N, )
        # -- we also need to make sure that the newly selected mean has distance at least D to every previous means --
        ok = False
        attempt = 0
        while not ok:
            if attempt >= (N-m):
                j = np.random.randint(0,N,1)
                break
            j = np.argmax(diff_sum) # the index of the pixel having the maximum difference to the previously selected means
            distances = np.sum(np.abs(mu[:m] - x[j]), axis=-1)
            ok = np.all(distances>D) # see if any previous mean has distance too close to the newly selected one
            if not ok:
                diff_sum[j] = -1
            attempt += 1
            
        mu[i] = x[j]
        
    cov = np.eye(3) * 0.1 # (3, 3)
    np.tile(cov[None,...], (K,1,1)) # (K, 3, 3)
    
    x_expanded = np.tile(x[:,None,:], (1,K,1)) # (N, K, 3)
    mu_expanded = np.tile(mu[None,...], (N,1,1)) # (N, K, 3)
    diff = np.sum(np.abs(x_expanded - mu_expanded), axis=-1) # (N, K)
    cluster_indexes = np.argmin(diff, axis=-1) # (N, ), the cluster index for each pixel
    cluster_indexes = np.tile(cluster_indexes[...,None], (1,K)).astype(np.uint32) # (N, K)
    comparator = np.arange(0,K,1).astype(np.uint32) # (K, )
    comparator = np.tile(comparator[None,...], (N,1)) # (N, K)
    association_mask = np.where(cluster_indexes==comparator, 1, 0)
    num_belonging = np.sum(association_mask, axis=0) # (K, )
    pi = num_belonging / N # (K, )
        
    return mu, cov, pi

def segmentation(c:np.ndarray, gmm_probs:np.ndarray, mu:np.ndarray):
    # c: (N, 3)
    # gmm_probs: (N, K)
    # mu: (K, 3)
    N, K = gmm_probs.shape
    
    seg = np.zeros_like(c)
    association = np.argmax(gmm_probs, axis=-1) # (N, )
    compare = np.tile(np.arange(0, K, 1)[None,...], (N,1)) # (N, K)
    association = np.tile(association[...,None], (1,K)) # (N, K)
    association_mask = np.where(compare==association, 1, 0) # (N, K)
    mu_extended = np.tile(mu[None,...], (N,1,1)) # (N, K, 3)
    seg = mu_extended[association_mask==1,...] # (N, 3)

    return seg
     
def GMM(image:np.ndarray, K:int, L:int):
    H, W, _ = image.shape
    c = image.reshape((-1,3)).astype(np.float32) / 255.0 # (N, 3)
    N = c.shape[0]
    
    # -- initializing K Gaussians --
    mu, cov, pi = cluster_init(c, N, K) # (K, 3), (K, 3, 3), (K, )
    print("==> Mu: \n", mu, "\n\n", "==> Cov: \n", cov, "\n\n", "==> Pi: \n", pi, "\n\n")
    
    # -- expectation maximization --
    for l in range(L):
        if l % 5 == 0:
            print("==> {}/{}".format(l,L))
        # -- compute the membership probabilities --
        pdfs = gaussian(c, mu, cov) # (N, K)
        pi = np.tile(pi[None,:], (N, 1)) # (N, K)
        p = pi * pdfs # (N, K)
        
        # -- segmentation --
        seg = segmentation(c, p, mu)
        seg = cv2.cvtColor((seg.reshape((H,W,3)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("s", seg)
        cv2.waitKey(10)
        
        p_sums = np.sum(p, axis=1, keepdims=True) # (N, K)
        p_normalized = p / p_sums # (N, K)
        # print("==> W: \n",  w, "\n\n")
        
        # -- update the weights --
        pi = np.mean(p_normalized, axis=0) # (K, )
        
        # -- compute pixel contribution --
        w = p_normalized / np.sum(p_normalized, axis=0, keepdims=True) # (N, K), the normalized pixel weights per Gaussian
        # print(np.sum(w, axis=0))
        w = np.tile(w[...,None], (1,1,3)) # (N, K, 3)
        c_extended = np.tile(c[:,None,:], (1,K,1)) # (N, K, 3)
        
        # -- update the means --
        mu = np.sum(w * c_extended, axis=0) # (K, 3)
        
        # -- update the covariances --
        mu_extended = np.tile(mu[None,...], (N,1,1)) # (N, K, 3)
        diff = (c_extended - mu_extended)[...,None] # (N, K, 3, 1)
        # print(c[W*15+16])
        # print(diff[W*15+16])
        # print(c[W*8+2])
        # print(diff[W*8+2])
        cov = diff @ np.transpose(diff, (0,1,3,2)) # (N, K, 3, 3)
        # print(cov[W*15+16])
        # print(cov[W*8+2])
        cov = np.tile(w[...,None], (1,1,1,3)) * cov # (N, K, 3, 3)
        cov = np.sum(cov, axis=0) # (K, 3, 3)
        for k in range(K):
            if np.linalg.matrix_rank(cov[k]) == 3:
                continue
            print("==> Singular covariance at {}".format(k))
            cov[k] = cov[k] + np.eye(3) * 0.15
        
        # print("==> Mu: \n", (mu*255).astype(np.uint8), "\n\n", "==> Cov: \n", cov, "\n\n", "==> Pi: \n", pi, "\n\n")
    # -- segmentation --
    seg = segmentation(c, p, mu)
    seg = cv2.cvtColor((seg.reshape((H,W,3)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return seg

fig = plt.figure()

seg = GMM(im, 5, 55)

seg = cv2.resize(seg, (W, H))
cv2.imwrite(image_file.replace("images/", "results/"), seg)
cv2.destroyAllWindows()