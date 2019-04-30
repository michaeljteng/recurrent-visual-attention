import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# glimpse network params
glimpse_arg = add_argument_group('Glimpse Network Params')
glimpse_arg.add_argument('--patch_size', type=int, default=64,
                         help='size of extracted patch at highest res')
glimpse_arg.add_argument('--glimpse_scale', type=int, default=1,
                         help='scale of successive patches')
glimpse_arg.add_argument('--num_patches', type=int, default=1,
                         help='# of downscaled patches per glimpse')
glimpse_arg.add_argument('--loc_hidden', type=int, default=128,
                         help='hidden size of loc fc')
glimpse_arg.add_argument('--glimpse_hidden', type=int, default=128,
                         help='hidden size of glimpse fc')
glimpse_arg.add_argument('--num_classes', type=int, default=10,
                       help='# of epochs to train for')


# core network params
core_arg = add_argument_group('Core Network Params')
core_arg.add_argument('--num_glimpses', type=int, default=5,
                      help='# of glimpses, i.e. BPTT iterations')
core_arg.add_argument('--hidden_size', type=int, default=256,
                      help='hidden size of rnn')


# reinforce params
reinforce_arg = add_argument_group('Reinforce Params')
reinforce_arg.add_argument('--std', type=float, default=0.17,
                           help='gaussian policy standard deviation')
reinforce_arg.add_argument('--constrain_mu', type=str2bool, default=True,
                           help='whether or not to perform tanh on policy mean')
reinforce_arg.add_argument('--M', type=float, default=10,
                           help='Monte Carlo sampling for valid and test sets')


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.1,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--batch_size', type=int, default=64,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')
data_arg.add_argument('--show_sample', type=str2bool, default=True,
                      help='Whether to visualize a sample grid of the data')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--attention_targets', type=str, default='approx', choices=['approx', 'exact'],
                       help='Whether to use targets for training the atteniton mechanism.')
train_arg.add_argument('--attention_target_weight', type=float, default=1.0,
                       help='Weighting given to attention targets in loss.')
train_arg.add_argument('--momentum', type=float, default=0.5,
                       help='Nesterov momentum value')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=3e-4,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=200,
                       help='Number of epochs to wait before stopping train')
train_arg.add_argument('--train_per_valid', type=int, default=10000,
                       help='Number of training traces to use between validations.')
train_arg.add_argument('--supervised_attention_prob', type=float, default=0.0,
                       help='Proportion of time to use targets for training the attention mechanism.')
train_arg.add_argument('--entropy_reinforce_loss', type=str2bool, default=False,
                       help='Use REINFORCE to force model to minimise entropy at each step.')


# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=False,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data',
                      help='Directory in which data is stored')
misc_arg.add_argument('--dataset', type=str, default='mnist',
                      help='Celeba or not', choices=['mnist', 'celeba', 'celebhq'])
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True,
                      help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--print_freq', type=int, default=10,
                      help='How frequently to print training details')
misc_arg.add_argument('--plot_freq', type=int, default=1,
                      help='How frequently to plot glimpses')
misc_arg.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
misc_arg.add_argument('--celebhq_image_dir', type=str, default='data/celebhq/images')
misc_arg.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
misc_arg.add_argument('--hq_attr_path', type=str, default='data/celebhq/annotations.p')
misc_arg.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
misc_arg.add_argument('--log_dir', type=str, default='stargan/logs')
misc_arg.add_argument('--model_save_dir', type=str, default='stargan/models')
misc_arg.add_argument('--sample_dir', type=str, default='stargan/samples')
misc_arg.add_argument('--result_dir', type=str, default='stargan/results')
misc_arg.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Male'])
misc_arg.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
misc_arg.add_argument('--image_size', type=int, default=28, help='image resolution')
misc_arg.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
