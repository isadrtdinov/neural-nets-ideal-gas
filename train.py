import time
from tqdm import tqdm
from data import random_crop
from entropy import EntropyLogger
from utility import *


def check_si_name(n, model_name):
    if model_name == 'resnet':
        return 'linear' not in n
    elif 'convnet' in model_name:
        return 'conv_layers' in n
    return False

def get_si_pnorm(model, model_name):
    """
    Get current model SI-pnorm
    """
    si_pnorm = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, model_name)))
    return si_pnorm

def fix_si_pnorm(model, si_pnorm_0, model_name):
    """
    Fix SI-pnorm to si_pnorm_0 value
    """
    si_pnorm = get_si_pnorm(model, model_name)
    p_coef = si_pnorm_0 / si_pnorm
    for n, p in model.named_parameters():
        if check_si_name(n, model_name):
            p.data *= p_coef


def train_model(model, criterion, optimizer, train_dataset, test_dataset, config, logfile, ckpt_path):
    trace_keys = [
        'weight', 'stoch_grad',  # save according to config.ckpt_iters
        'train_loss', 'train_acc', 'test_loss', 'test_acc',  # save according to config.log_iters
        'snr_new', 'snr_old', 'full_grad_norm', 'noise_mean_norm', 'stoch_grad_rms',
        'stoch_grad_mean_norm', 'stoch_grad_mean_norm_squared',
        'spherical_entropy', 'log_radius', 'mean_radius',
        'running_loss', 'running_radius', 'stoch_grad_norm'  # save each running_log_freq iterations
    ]
    trace = {key: [] for key in trace_keys}
    entropy_logger = EntropyLogger(queue_size=config.queue_size)

    if logfile:
        with open(logfile, "a") as file:
            print("Start training", file=file)
            start_time = time.time()

    if config.use_wandb:
        import wandb
        if config.opt_mode == 'fixed_sphere':
            name = f'{config.model_name}{config.num_channels}_lr{config.lr:.2e}_r{config.si_pnorm_0:.2e}'
        else:
            name = f'{config.model_name}{config.num_channels}_lr{config.lr:.2e}_wd{config.wd:.2e}'
        wandb.init(project='SI-thermodynamics', config=config, group=config.opt_mode, name=name)

    def log(iteration):
        log_dict = {}
        metrics, full_grad = eval_model(model, criterion, train_dataset, test_dataset)
        for key, value in metrics.items():
            trace[key].append(value)
            log_dict[key] = value

        stoch_grads = get_stoch_grads(model, criterion, train_dataset,
                                      config.queue_size, config.batch_size, requires_grad=True)
        snr_dict = get_snr(full_grad, stoch_grads)
        for key, value in snr_dict.items():
            trace[key].append(value)
            log_dict[key] = value

        spherical_entropy, log_radius = entropy_logger.get_metrics()
        mean_radius = entropy_logger.get_radius()
        trace['spherical_entropy'].append(spherical_entropy)
        trace['log_radius'].append(log_radius)
        trace['mean_radius'].append(mean_radius)

        log_dict['spherical_entropy'] = spherical_entropy
        log_dict['log_radius'] = log_radius
        log_dict['mean_radius'] = mean_radius
        log_dict['iteration'] = iteration
        if config.use_wandb:
            wandb.log(log_dict)

    if 0 in config.log_iters:
        log(iteration=0)

    if 0 in config.ckpt_iters:
        weights = parameters_to_vector(model.parameters())
        trace['weight'].append(weights.cpu())
        trace['stoch_grad'].append(torch.full_like(weights, fill_value=torch.nan))

    if not config.iid_batches:
        cur_i = 0
        iid = torch.randperm(len(train_dataset))

    si_pnorm = get_si_pnorm(model, config.model_name)
    for it in tqdm(range(1, config.num_iters + 1)):
        optimizer.zero_grad(set_to_none=True)
        if config.iid_batches:
            iid = torch.randint(0, len(train_dataset), (config.batch_size,))
            input, target = train_dataset[iid]
        else:
            j = min(cur_i + config.batch_size, len(train_dataset))
            input, target = train_dataset[iid[cur_i:j]]
            if cur_i + config.batch_size >= len(train_dataset):
                cur_i = 0
                iid = torch.randperm(len(train_dataset))
            else:
                cur_i += config.batch_size

        if config.augment:
            input = random_crop(input)

        model.train()
        if config.opt_mode == 'fixed_elr':
            optimizer.param_groups[0]['lr'] = config.lr * si_pnorm ** 2
        elif config.opt_mode == 'fixed_sphere':
            optimizer.param_groups[0]['lr'] = config.lr * config.si_pnorm_0 ** 2

        loss, output = criterion(model, input, target, "mean")
        loss.backward()
        optimizer.step()

        si_pnorm = get_si_pnorm(model, config.model_name)
        if config.opt_mode == 'fixed_sphere':
            fix_si_pnorm(model, config.si_pnorm_0, config.model_name)

        stoch_grad = get_model_gradients(model, requires_grad=True).cpu()
        if it % config.queue_freq == 0:
            entropy_logger.add_weights(get_model_parameters(model, requires_grad=True).detach().cpu())

        if it % config.running_log_freq == 0:
            trace['running_loss'].append(loss.item())
            trace['running_radius'].append(si_pnorm)
            trace['stoch_grad_norm'].append(stoch_grad.norm().item())

        if it in config.log_iters:
            log(iteration=it)

            if logfile:
                with open(logfile, "a") as file:
                    cur_time = time.time()
                    print(
                        f'Elapsed: {cur_time - start_time:.2f}s. '
                        f'Logged metrics at iteration {it}', file=file
                    )

        if it in config.ckpt_iters:
            weights = parameters_to_vector(model.parameters()).cpu()
            trace['weight'].append(weights)
            trace['stoch_grad'].append(stoch_grad)

            torch.save({
                'model': model.state_dict(),
                'trace': trace
            }, ckpt_path)

            if logfile:
                with open(logfile, "a") as file:
                    cur_time = time.time()
                    print(
                        f'Elapsed: {cur_time - start_time:.2f}s. '
                        f'Checkpointed weights and grads at iteration {it}', file=file
                    )

    return trace
