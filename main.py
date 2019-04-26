import torch

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from data_loader import get_gen_model_loader, \
    get_partially_supervised_attention_loader
from experiment_utils import track_metadata, save_details


@track_metadata
def main(config, get_metadata=None):

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    # instantiate data loaders
    if config.is_train:
        train_loader = get_partially_supervised_attention_loader(
            config.batch_size,
            config.train_per_valid,
            config.supervised_attention_prob,
            **kwargs
        )
        valid_loader = get_gen_model_loader(
            config.batch_size,
            epoch_size=config.valid_size,
            fix_data=True,
            **kwargs
        )
        data_loader = (train_loader, valid_loader)
    else:
        data_loader = get_gen_model_loader(
            config.batch_size,
            epoch_size=10000,
            fix_data=True,
            fix_offset=1e6,
            **kwargs
        )

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        losses = trainer.train()
        save_details(config, get_metadata(), losses)

    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
