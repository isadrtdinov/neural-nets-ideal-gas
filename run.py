import argparse
import numpy as np
from types import SimpleNamespace
from scripts import pretrain, evaluate_on_checkpoints


def get_config(seed, device, num_iters, lr, wd, si_pnorm_0, opt_mode, num_channels, model_name, dataset='cifar10',
               samples_per_label=5000, loss_fn='ce', regime="pretrain"):
    if regime == "pretrain":
        log_iters = np.unique(np.array(
            np.arange(200, 10_000, 200).tolist() + np.logspace(4, np.log10(num_iters), 151).tolist()
        ).astype(int))
        ckpt_iters = np.array([0] + np.logspace(4, np.log10(num_iters), 9).tolist()).astype(int)
    elif regime == "ckpt":
        if num_iters is None:
            num_iters = 1000
        log_iters = np.array([num_iters])
        ckpt_iters = np.array([])

    dataset_size = samples_per_label * (10 if dataset == 'cifar10' else 100)
    if opt_mode == 'fixed_sphere':
        savedir = f'experiments/{model_name}{num_channels}_{dataset}_{dataset_size}obj_{loss_fn}/'\
                  f'{opt_mode}/lr{lr:.2e}_r{si_pnorm_0:.2e}_seed{seed}'
    else:
        savedir = f'experiments/{model_name}{num_channels}_{dataset}_{dataset_size}obj_{loss_fn}/'\
                  f'{opt_mode}/lr{lr:.2e}_wd{wd:.2e}_seed{seed}'

    return SimpleNamespace(
        dataset=dataset, # 'cifar10 or cifar100
        data_path='./datasets',  # path to datasets
        augment=False,  # whether to use augmentations (random crop in case of CIFAR-10/100)
        model_name=model_name,  # convnet or resnet
        samples_per_label=samples_per_label,  # size of the dataset / 10 for CIFAR-10; default is 5000
        batch_size=128,  # training batch size
        iid_batches=True,  # sample batches independently or train by epochs
        si_pnorm_0=si_pnorm_0,  # parameter norm (initial radius for training in full space)
        num_channels=num_channels,  # number of channels in ConvNet
        last_layer_norm=10.0 if loss_fn == 'ce' else 1.5,  # last layer norm in MLP (last layer is fixed and not trained)
        dtype='float32',  # training data type
        device=device,  # computational device
        loss_fn=loss_fn,  # loss function used for training (ce or mse)
        num_iters=num_iters,  # pre-training iterations
        ckpt_iters=ckpt_iters.tolist(),  # how often to checkpoint the model
        log_iters=log_iters.tolist(),  # how often to calculate metrics
        running_log_freq=50,  # how often to log running metrics (e.g., batch loss)
        pt_seed=seed,  # training seed
        data_seed=1,  # dataset generation seed
        init_point_seed=1,  # random init seed
        last_layer_seed=4,  # classifier head initialization seed
        savedir=savedir,  # pre-training path
        lr=lr,  # learning rate range to use
        wd=wd,  # weight decay coefficient
        opt_mode=opt_mode,  # train with a fixed LR, ELR or on a sphere
        queue_size=4000,  # queue size for entropy calculation and snr estimation
        queue_freq=25,  # how often to add weights to entropy queue
        use_wandb=True  # whether to log to wandb
    )


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float)
parser.add_argument('--wd', type=float)
parser.add_argument('--si_pnorm_0', type=float, default=1.0)
parser.add_argument('--opt_mode', type=str, default='fixed_lr',
                    choices=['fixed_lr', 'fixed_elr', 'fixed_sphere'])
parser.add_argument('--seeds', nargs="+", type=int)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--samples_per_label', type=int, default=5000)
parser.add_argument('--loss_fn', type=str, default="ce")
parser.add_argument('--regime', type=str, default="pretrain") # pretrain or ckpt
parser.add_argument('--model', type=str, default='convnet', choices=['convnet', 'resnet'])
parser.add_argument('--num_iters', type=int, default=1_000_000)
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()

assert args.lr is not None
if args.opt_mode != 'fixed_sphere':
    assert args.wd is not None and args.wd > 0
else:
    args.wd = 0  # set wd = 0 for training on a sphere

for seed in args.seeds:
    config = get_config(seed=seed, device=args.device, num_iters=args.num_iters, lr=args.lr, wd=args.wd, si_pnorm_0=args.si_pnorm_0,
                        opt_mode=args.opt_mode, num_channels=args.channels, model_name=args.model, dataset=args.dataset,
                        samples_per_label=args.samples_per_label, loss_fn=args.loss_fn, regime=args.regime)
    if args.regime == "pretrain":
        pretrain(config)
    else:
        evaluate_on_checkpoints(config)
