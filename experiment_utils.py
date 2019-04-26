import pickle
import time
import git


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


def save_details(config, metadata, losses):
    if config.path is not None:
        filename = "{}/{}_{}_{}_{}.p".format(
            config.path,
            config.num_glimpses,
            config.supervised_attention_prob,
            config.random_seed,
            str(metadata['start_time'])[-5:]
        )
        pickle.dump({"config": config.__dict__,
                     "metadata": metadata,
                     "losses": losses},
                    open(filename, 'wb'))
