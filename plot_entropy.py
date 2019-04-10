import matplotlib.pyplot as plt
import numpy as np
import torch
MIN_X, MAX_X = 0, 224-16
attributes = ['5 o Clock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes',
              'Bald', 'Bangs', 'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair',
              'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby', 'Double Chin',
              'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones',
              'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard',
              'Oval Face', 'Pale Skin', 'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks',
              'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair', 'Wearing Earrings',
              'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace', 'Wearing Necktie',
              'Young']

def plot_entropy(APEs, attr_i=None, path=None):
    coords = [int(x) for x in np.linspace(MIN_X, MAX_X, 50)]
    Z = np.array([[APEs[(y, x)][attr_i] if attr_i is not None else APEs[(y, x)] for x in coords] for y in coords])
    print('attribute plot for: ', attributes[attr_i])
    #  import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    im = ax.imshow(Z)
    fig.colorbar(im)
    plt.title(attributes[attr_i])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.show()
	
import pickle
APEs = pickle.load(open("data_ALL_CLASSES_16x16_RESUMED_AGAIN_weights_64_BEST_100_APEs.p", 'rb'))
plot_entropy(APEs, attr_i=20)
