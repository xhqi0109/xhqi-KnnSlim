import os
import copy
import logging
import numpy as np
import time
import torch
from thop import profile
from thop import clever_format
import warnings
import yaml

import sys
from xhqi_knnslim.utils import enums

# Define a function to ensure x is a list
def tolist(x):
    """
    x to list
    """
    if not isinstance(x, list):
        x = [x]
    return x


# Generate YAML configuration file from JSON
def generate_yaml(weights, info, prune_ratios, filename):
    """
    write config information from json to yaml
    @param weights: dict {name, param}
    @param info: some config parameters
    @param prune_ratios: ratios to be written
    @param filename: yaml file to write parameters
    return:
        yaml file name
    """
    filename = filename[:-4] + 'yaml'
    with open(filename, 'w', encoding='utf-8') as f:
        # Write configurations
        for key, value in info.items():
            f.write(f'{key} : \n  {value}\n')

        f.write('\n')
        f.write('# weight :  # shape=(out_channel, input_channel, kernel, kernel)\n')
        f.write('#   rate\n')
        f.write('\n')
        # Write weight pruning ratios
        f.write('prune_ratios:\n')
        for name, var in weights.items():
            if name in prune_ratios.keys():
                f.write(f'  {name} :    # {var.shape}\n')
                f.write(f'    {prune_ratios[name]}\n')
            else:
                f.write(f'#  {name} :    # {var.shape}\n')
    print(f'-> prune_cfg saved at {filename}')
    return filename


# Get the type of layer in a torch module
def get_type(layer):
    """
    get the layer type of one torch module
    """
    if 'Conv2d' in layer._get_name():
        if layer.groups == layer.out_channels and layer.out_channels > 1:
            if 'Quant' in layer._get_name():
                return 'DepthwiseConv2DQuant'
            else:
                return 'DepthwiseConv2D'
        else:
            if 'Quant' in layer._get_name():
                return layer._get_name()
            else:
                return 'Conv2D'
    else:
        return layer._get_name()


# Assign numpy value y to Tensor x
def assign_tensor(x, y):
    """
    assign the numpy value of y to x (Tensor)
    """
    assert x.shape == y.shape
    assert isinstance(x, torch.Tensor)
    if isinstance(y, np.ndarray):
        z = y
    elif isinstance(y, torch.Tensor):
        if y.is_cuda:
            z = y.detach().cpu().numpy()
        else:
            z = y.detach().numpy()
    else:
        raise ValueError('Unknown y of type', y.type)
    if x.is_cuda:
        x.data = torch.from_numpy(z).cuda()
    else:
        x.data = torch.from_numpy(z)

    return x


# Calculate the number of parameters and multi-add float operations of a model
def torch_summary(model, input_tuple, verbose=False):
    """
    calculate the number of parameters and multi-add float operations
    """
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        cpu_model = copy.deepcopy(model.module).to('cpu')
    else:
        cpu_model = copy.deepcopy(model).to('cpu')
    # Calculate FLOPs and parameters
    flops, params = profile(cpu_model, input_tuple, verbose=verbose)

    del cpu_model
    return params, flops


# Get a summary of pruning information
def get_prune_summary(ori_model, small_model, inputs_tuple, logger):
    logger.info(f'{"param_name":30}{"original_shape":30}{"compress_shape":30}')
    logger.info('-' * 90)
    small_model_weights = dict(small_model.named_parameters())
    
    for name, param in ori_model.named_parameters():
        origin_shape = str(param.shape)
        prune_shape = str(small_model_weights[name].shape)
        logger.info(f'{name:30}{"  ====  "}{origin_shape:30}{prune_shape:30}')
    
    origin_params, origin_flops = torch_summary(ori_model, inputs_tuple)
    prune_params, prune_flops = torch_summary(small_model, inputs_tuple)
    compress_ratio = (1 - prune_params / origin_params) * 100
    origin_params_str, origin_flops_str, prune_params_str, prune_flops_str = clever_format(
        [origin_params, origin_flops, prune_params, prune_flops], '%.4f')
    logger.info(f'[Origin Model] Params: {origin_params_str}, Flops: {origin_flops_str}\n'
                f'[ Slim Model ] Params: {prune_params_str}, Flops: {prune_flops_str}\n'
                f'\ncompress ratio: {compress_ratio:.2f} %')


# Get the number of devices
def get_num_device(model):
    param = next(model.named_parameters())
    if str(param[1].device) == 'cpu':
        return 0
    elif 'cuda' in str(param[1].device):
        if param[0].startswith('module.'):
            return torch.cuda.device_count()
        else:
            return 1
    return None


# Prefix name with 'module.' if using multiple devices
def prefix(name, device=1):
    assert isinstance(name, str)
    return name
    if device > 1:
        if not name.startswith('module.'):
            name = 'module.' + name
    elif device <= 1:
        if name.startswith('module.'):
            name = name.split('module.')[-1]
    return name


# Update list x with y and remove duplicates
def list_update(x, y):
    if not isinstance(x, (list, tuple)):
        x = tolist(x)
    if not isinstance(y, (list, tuple)):
        y = tolist(y)
    z = list(x) + list(y)
    z = sorted(set(z), key=z.index)
    return z


# Get logger instance
def get_logger(pid, stage):
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    log_name = f'./logs/pid_{pid}_{stage}_{time.strftime("%Y-%m-%d-%H-%M")}.log'

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # Handler
    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(fmt='[%(asctime)s] - [line:%(lineno)d] - %(levelname)s => %(message)s',
                                  datefmt='%Y-%m-%d %H:%M')

    # add formatter to handler
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f'Creates a log: {log_name}')

    return logger


# Load extra dependencies from a YAML file
def load_extra_dependencies(yaml_file, num_devices):
    """
    Parse the extra_dependencies file, to handle add or concat operations
    Args:
        yaml_file: path to the extra_dependencies file
    """
    dependencies = {'delete': {}, 'append': {}, 'add': {}, 'cat': {}, 'add_prev_set': set(), 'cat_prev_set': set()}
    if yaml_file is None:
        print('-> NO extra dependency is used.')
        return dependencies
    if not os.path.exists(yaml_file):
        warnings.warn(f'-> Yaml "{yaml_file}" NOT found')
        return dependencies
    with open(yaml_file, 'r', encoding='utf-8') as stream:
        try:
            raw_dict = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
        if not raw_dict:
            warnings.warn(f'-> "{yaml_file}" is empty')
            return dependencies
        for key, value in raw_dict.items():
            if key in ['delete_dependencies', 'append_dependencies', 'add_dependencies', 'cat_dependencies']:
                dependencies[key.split('_')[0]] = raw_dict[key]

        temp_dict = {}
        for key, value in dependencies['delete'].items():
            k = prefix(key, num_devices)
            vs = [prefix(v, num_devices) for v in tolist(value)]
            temp_dict[k] = vs
        dependencies['delete'] = temp_dict

        temp_dict = {}
        for key, value in dependencies['append'].items():
            k = prefix(key, num_devices)
            vs = [prefix(v, num_devices) for v in tolist(value)]
            temp_dict[k] = vs
        dependencies['append'] = temp_dict

        temp_dict = {}
        for name, info in dependencies['add'].items():
            prevs, nexts = info.values()
            prevs = [prefix(name, num_devices) for name in tolist(prevs)]
            nexts = list(filter(None, tolist(nexts)))
            nexts = [prefix(name, num_devices) for name in tolist(nexts)]
            temp_dict[tuple(prevs)] = [tuple([name]), tuple(nexts)]
            dependencies['add_prev_set'].update(prevs)
        dependencies['add'] = temp_dict

        temp_dict = {}
        for name, info in dependencies['cat'].items():
            prevs, nexts = info.values()
            prevs = [prefix(name, num_devices) for name in tolist(prevs)]
            nexts = list(filter(None, tolist(nexts)))
            nexts = [prefix(name, num_devices) for name in tolist(nexts)]
            temp_dict[tuple(prevs)] = [tuple([name]), tuple(nexts)]
            dependencies['cat_prev_set'].update(prevs)
        dependencies['cat'] = temp_dict
    return dependencies


# Merge two dependency dictionaries
def merge_dependencies(dict1, dict2):
    """merge the attributes of two dicts.
    keys: delete, append, add, cat, add_set, cat_set

    Return:
        merged dict
    """
    assert 'delete' in dict1 and 'delete' in dict2
    dict1['delete'].update(dict2['delete'])

    assert 'append' in dict1 and 'append' in dict2
    dict1['append'].update(dict2['append'])

    assert 'add' in dict1 and 'add' in dict2
    if not dict1['add']:
        dict1['add'] = dict2['add']
    else:
        for key in dict2['add']:
            if key not in dict1['add']:
                warnings.warn(f'Auto dependency v.s. Extra dependency.\n'
                              f'{key} NOT found in add {dict1["add"].keys()}.')
                dict1['add'][key] = dict2['add'][key]
            else:
                add_names = list_update(dict1['add'][key][0], dict2['add'][key][0])
                next_names = list_update(dict1['add'][key][1], dict2['add'][key][1])
                extra_names = set(next_names) - set(dict1['add'][key][1])
                if extra_names:
                    warnings.warn(f'Auto dependency v.s. Extra dependency.\n'
                                  f'Extra {extra_names} found in next {dict2["add"][key][1]} '
                                  f'of {dict2["add"][key][0]} prev {key}.')
                dict1['add'][key][0] = tuple(add_names)
                dict1['add'][key][1] = tuple(next_names)

    assert 'cat' in dict1 and 'cat' in dict2
    if not dict1['cat']:
        dict1['cat'] = dict2['cat']
    else:
        for key in dict2['cat']:
            if key not in dict1['cat']:
                warnings.warn(f'Auto dependency v.s. Extra dependency.\n'
                              f'{key} NOT found in cat {dict1["cat"].keys()}.')
                dict1['cat'][key] = dict2['cat'][key]
            else:
                cat_names = list_update(dict1['cat'][key][0], dict2['cat'][key][0])
                next_names = list_update(dict1['cat'][key][1], dict2['cat'][key][1])
                extra_names = set(next_names) - set(dict1['cat'][key][1])
                if extra_names:
                    warnings.warn(f'Auto dependency v.s. Extra dependency.\n'
                                  f'Extra {extra_names} found in next {dict2["cat"][key][1]} '
                                  f'of {dict2["cat"][key][0]} prev {key}.')
                dict1['cat'][key][0] = tuple(cat_names)
                dict1['cat'][key][1] = tuple(next_names)

    assert 'add_prev_set' in dict1 and 'add_prev_set' in dict2
    new_set = dict1['add_prev_set'] | dict2['add_prev_set']
    extra_names = new_set - dict1['add_prev_set']
    if dict1['add_prev_set'] and extra_names:
        warnings.warn(f'Auto dependency v.s. Extra dependency.\n'
                      f'extra {extra_names} found in add_set {dict2["add_set"]}')
    dict1['add_prev_set'] = new_set

    assert 'cat_prev_set' in dict1 and 'cat_prev_set' in dict2
    new_set = dict1['cat_prev_set'] | dict2['cat_prev_set']
    extra_names = new_set - dict1['cat_prev_set']
    if dict1['cat_prev_set'] and extra_names:
        warnings.warn(f'Auto dependency v.s. Extra dependency.\n'
                      f'extra {extra_names} found in cat_set {dict2["cat_set"]}')
    dict1['cat_prev_set'] = new_set

    return dict1


# Revise weight dependencies based on 'delete', 'append', and 'exclude'
def revise_dependencies(d_dict, exclude_layer_pruning_ratio):
    """modify the weight dependencies according to the 'delete', 'append', and 'exclude'"""

    for key, value in d_dict['delete'].items():
        if key in d_dict['weight']:
            tmp_list = list(d_dict['weight'][key])
            for name in value:
                tmp_list.remove(name)
            d_dict['weight'][key] = tuple(tmp_list)

    for key, value in d_dict['append'].items():
        if key in d_dict['weight']:
            tmp_list = list(d_dict['weight'][key])
            for name in value:
                tmp_list.append(name)
            d_dict['weight'][key] = tuple(tmp_list)

    for name in exclude_layer_pruning_ratio:
        if name in d_dict['weight']:
            d_dict['weight'].pop(name)

    d_dict_copy = copy.deepcopy(d_dict)
    for name in exclude_layer_pruning_ratio:
        for key in d_dict['add']:
            if name in key and key in d_dict_copy['add']:
                d_dict_copy['add'].pop(key)
        for key in d_dict['cat']:
            if name in key and key in d_dict_copy['cat']:
                d_dict_copy['cat'].pop(key)
    d_dict = d_dict_copy
    return d_dict


# Detach cat-add dependencies from cat operations
def detach_cat_add_dependencies(dependencies):
    """Detach dependencies related to cat-add opertions from cat operations.
    Args:
        dependencies(dict): Gained dependencies from model analyzer module.
    """
    dependencies['cat-add'] = {}  # cat-add算子的前后依赖
    dependencies['cat-add_set'] = set()  # 后依赖有add的cat算子集合
    dependencies['cat-add_prev_set'] = set()  # cat-add前依赖的算子集合
    for weight in dependencies['add_prev_set']:
        if 'cat' in weight:  # 如果是cat算子
            dependencies['cat-add_set'].add(weight)
            dependencies['cat-add_prev_set'].update(dependencies['cat_prev_dict'][weight])
            dependencies['cat_prev_set'].difference_update(dependencies['cat_prev_dict'][weight])
    for prev_cat, (cat_list, nexts_cat) in list(dependencies['cat'].items()):
        for cat_op in cat_list:
            if cat_op in dependencies['add_prev_set']:
                dependencies['cat'].pop(prev_cat)
                dependencies['cat-add'][prev_cat] = [cat_list, nexts_cat]
                continue
    return dependencies


# Gain cat-prev dependencies
def gain_cat_prev_dependencies(dependencies):
    dependencies['cat_prev_dict'] = {}  # cat_op: its prev weights
    for prev_weights, (cat_names, _) in dependencies['cat'].items():
        for single_cat_name in cat_names:
            dependencies['cat_prev_dict'][single_cat_name] = prev_weights
    return dependencies


# Save model checkpoint
def save_checkpoint(admm_handler, epoch, optimizer, ckpt_name):
    checkpoint = {
        'epoch': epoch,
        'model': admm_handler.model,
        'state_dict': admm_handler.model.state_dict(),
        'optimizer': optimizer,
        'admm_z': admm_handler.admm_z,
        'admm_u': admm_handler.admm_u,
    }
    admm_handler.logger.info(f'Save model name:\t{ckpt_name}')
    torch.save(checkpoint, ckpt_name)


# Load pretrained model
def load_pretrained_model(admm_handler):
    """load model state dict on the beginning of per stage
    Args:
        admm_handler: admm pruner
    """
    assert admm_handler.pretrained_state is not None, 'Needs a pretrained model checkpoint'
    if not os.path.exists(admm_handler.pretrained_state):
        raise Exception(f'Not existing pretrained checkpoint {admm_handler.pretrained_state}')

    admm_handler.logger.info(f'Load model from {admm_handler.pretrained_state} '
                                f'for {str(admm_handler.stage).rsplit(".", maxsplit=1)[-1]} Stage')
    checkpoint = torch.load(admm_handler.pretrained_state, map_location=admm_handler.device)
    if "state_dict" in checkpoint.keys():
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    state_dict = {k.replace('module.', '') if admm_handler.num_devices <= 1 else k: v for k, v in state_dict.items()}
    
    admm_handler.model.load_state_dict(state_dict, strict=True)


# Load breakpoint for admm
def load_breakpoint(admm_handler):
    """load saved admm parameters in interrupted admm traing process
    Args:
        admm_handler: admm pruner
    """
    assert admm_handler.pretrained_state is not None, 'No Interrupted admm checkpoint'
    start_epoch = 0
    admm_handler.logger.info(f'Loading from interrupted admm checkpoint {admm_handler.pretrained_state}')
    checkpoint = torch.load(admm_handler.pretrained_state, map_location=admm_handler.device)
    start_epoch = checkpoint['epoch'] + 1
    admm_handler.logger.info(f'after resume, start epoch is {start_epoch}')
    optimizer = checkpoint['optimizer']
    admm_handler.model = checkpoint['model'].to(admm_handler.device)

    if admm_handler.stage == enums.PruningStage.PRUNE:
        admm_handler.admm_z = checkpoint['admm_z']
        admm_handler.admm_u = checkpoint['admm_u']
    admm_handler.logger.info('Loading Checkpoint Finished!')

    admm_handler.start_epoch = start_epoch
    admm_handler.optimizer = optimizer


# Publish model
def publish(admm_handler, ckpt_name):
    torch.save(admm_handler.model.state_dict(), ckpt_name)
