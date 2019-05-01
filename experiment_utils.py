import pickle
import time
import git
import numpy as np
import torch


def track_metadata(foo):
    start = time.time()
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    def get_metadata():
        return {"start_time": start,
                "finish_time": time.time(),
                "git_hash": sha}

    def tracked(*args, **kwargs):
        return foo(*args, **kwargs,
                   get_metadata=get_metadata)

    return tracked


def save_details(config, metadata, losses, model_name):
    if config.path is not None:
        filename = "{}/{}_{}_{}.p".format(
            config.path,
            model_name, 
            config.random_seed,
            str(metadata['start_time'])[-5:]
        )
        pickle.dump({"config": config.__dict__,
                     "metadata": metadata,
                     "losses": losses},
                    open(filename, 'wb'))

class ApproxAttention(object):
    def __init__(self, attribute, i_size, g_size, cutoff=0.8):
        attr_list = ['5 o Clock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes',
                      'Bald', 'Bangs', 'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair',
                      'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby', 'Double Chin',
                      'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones',
                      'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard',
                      'Oval Face', 'Pale Skin', 'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks',
                      'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair', 'Wearing Earrings',
                      'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace', 'Wearing Necktie',
                      'Young']
        assert attribute in attr_list
        assert i_size % g_size == 0
        attr_i = attr_list.index(attribute)
        MIN_X, MAX_X = 0, i_size - g_size
        self.num_glimpses = int(i_size / g_size)**2
        coords = [int(x) for x in np.linspace(MIN_X, MAX_X, 50)]
        APEs = pickle.load(open("data_ALL_CLASSES_16x16_RESUMED_AGAIN_weights_64_BEST_100_APEs.p", 'rb'))
        self.Z = 1 - np.array([[APEs[(y, x)][attr_i] \
                                if attr_i is not None \
                                else APEs[(y, x)] \
                            for x in coords] for y in coords])
        maxZ = np.max(self.Z)
        entropy_cutoff = 0.95 * maxZ
        self.sample_probs = torch.softmax(
                                torch.tensor(
                                    np.where(self.Z > entropy_cutoff, 
                                            self.Z, 
                                            np.zeros((50,50)) - np.inf
                                        ).flatten()
                                ), 0)
        self.sampler = None

    def sample(self, batch_size, num_glimpses):
        #  print('wtf')
        #  if self.sampler == None:
        self.sampler = torch.distributions.Categorical(self.sample_probs.expand(batch_size, num_glimpses, self.sample_probs.shape[0]))
        unnormed = self.sampler.sample()
        return torch.round((unnormed.float() / 2500)*self.num_glimpses).long()
        

