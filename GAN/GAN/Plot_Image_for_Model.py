import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

G = models.load_model('generator_model.h5')

width = 28
height = 28
z_dim = 100


def Plot_Generated(n_ex = 5, dim = (1, 5), figsize = (12, 2)):
    noise = np.random.normal(0, 1, size = (n_ex, z_dim))
    generated_images = G.predict(noise)
    generated_images = generated_images.reshape(generated_images.shape[0], width, height)
    plt.figure(figsize = figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :], interpolation = 'nearest', cmap = 'gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
Plot_Generated()
