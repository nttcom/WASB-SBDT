import torch.optim as optim

from .lr_schedulers import MultiStepLRWithWarmUp

__optimizer_factory = {
        'adam': optim.Adam,
        'adadelta' : optim.Adadelta,
        'sgd': optim.SGD,
        }

__scheduler_factory = {
        'constant': optim.lr_scheduler.ConstantLR,
        'multistep': optim.lr_scheduler.MultiStepLR,
        'multistep_warmup': MultiStepLRWithWarmUp,
        }

def build_optimizer_and_scheduler(cfg, model_parameters):
    # optimizer 
    optimizer_name = cfg['optimizer']['name']
    if not optimizer_name in __optimizer_factory.keys():
        raise KeyError('invalid optimizer: {}'.format(optimizer_name ))
    if optimizer_name=='adam':
        lr           = cfg['optimizer']['learning_rate']
        weight_decay = cfg['optimizer']['weight_decay']
        optimizer    = __optimizer_factory[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name=='adadelta':
        lr           = cfg['optimizer']['learning_rate']
        weight_decay = cfg['optimizer']['weight_decay']
        optimizer    = __optimizer_factory[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name=='sgd':
        lr           = cfg['optimizer']['learning_rate']
        momentum     = cfg['optimizer']['momentum']
        weight_decay = cfg['optimizer']['weight_decay']
        optimizer = __optimizer_factory[optimizer_name](model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # scheduler
    scheduler_name = cfg['optimizer']['scheduler']['name']
    if not scheduler_name in __scheduler_factory.keys():
        raise KeyError('invalid scheduler: {}'.format(scheduler_name))
    if scheduler_name in ['multistep', 'multistep_warmup']:
        milestones = cfg['optimizer']['scheduler']['stepsize']
        gamma      = cfg['optimizer']['scheduler']['gamma']
        scheduler  = __scheduler_factory[scheduler_name](optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name in ['constant']:
        factor    = cfg['optimizer']['scheduler']['factor']
        scheduler = __scheduler_factory[scheduler_name](optimizer, factor=factor)
    return optimizer, scheduler

