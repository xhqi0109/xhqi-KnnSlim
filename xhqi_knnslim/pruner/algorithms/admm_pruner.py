from .metapruner import MetaPruner

import json
import sys

import torch

from xhqi_knnslim.callback import pruner_callback
from xhqi_knnslim.utils import enums
from xhqi_knnslim.utils import pruner_utils
import torch
import torch.nn as nn
import typing, warnings

from .scheduler import linear_scheduler
from ..import function
from ..import importance
from ... import ops, dependency, _helpers

class AdmmPruner(MetaPruner):
    """ 
    A class representing an ADMM pruner.
    
    Args:
        model (nn.Module): A PyTorch model.
        config: Configuration settings for ADMM.
        stage (str): Stage of ADMM training.
        inputs_tuple (torch.Tensor): Dummy input for graph tracing.
        pid (int): Process ID.
        load_model (bool): Whether to load the model.
        resume (bool): Whether to resume training.
        pretrained_state: Pretrained model state.
        external_init_lr: External initial learning rate.
        ckpt_dir (str): Directory to save checkpoints.
        ch_sparsity_dict (dict): Layer-specific sparsity dictionary.
        ignored_layers (list): Ignored layers list.

    Raises:
        ValueError: If invalid configuration settings are provided.
    """
    def __init__(
        self,
        model: nn.Module,
        config,        
        stage,
        inputs_tuple: torch.Tensor,
        pid=0, 
        load_model=False, 
        resume=False, 
        pretrained_state=None,
        external_init_lr=None, 
        ckpt_dir="./checkpoints",
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
        ignored_layers: typing.List[nn.Module] = None,
    ):

        json_cfg = self._load_config(config)
        ch_sparsity = json_cfg['ch_sparsity']
        if not isinstance(ch_sparsity, (int, float)) or ch_sparsity < 0 or ch_sparsity > 1:
            raise ValueError("ch_sparsity must be a float between 0 and 1")

        max_ch_sparsity = json_cfg["max_ch_sparsity"]
        if not isinstance(max_ch_sparsity, (int, float)) or max_ch_sparsity < 0 or max_ch_sparsity > 1:
            raise ValueError("max_ch_sparsity must be a float between 0 and 1")

        round_to = json_cfg["round_to"]
        if not isinstance(round_to, (int)):
            raise ValueError("round_to must be an integer")

        if not isinstance(pid, (int)):
            raise ValueError("pid must be an integer")
        self.pid = pid
        
        stage = enums.str2enums_id(stage)
        if not stage:
            raise ValueError("stage cannot be None")
        self.stage = stage

        self.ckpt_dir = ckpt_dir
        self.load_model = load_model
        self.resume = resume
        self.pretrained_state = pretrained_state
        self.rho = json_cfg['rho']
        self.admm_epoch = json_cfg['admm_epoch']
        self.internal_stage = json_cfg['internal_stage']
        self.init_lr = json_cfg['init_lr'] if external_init_lr is None else external_init_lr
        self.lr_decay = json_cfg['lr_decay']

        self.admm_u = {}
        self.admm_z = {}

        self.logger = pruner_utils.get_logger(self.pid, self.stage)
        self.num_devices = pruner_utils.get_num_device(model)
        self.device = torch.device('cpu' if self.num_devices == 0 else 'cuda')
        self.logger.info(f'------------- ADMM {self.stage} -------------')

        self.start_epoch, self.optimizer = 0, None

        imp = importance.MagnitudeImportance()

        super(AdmmPruner, self).__init__(
            model=model,
            example_inputs=inputs_tuple,
            importance=imp,
            iterative_steps=1,
            iterative_sparsity_scheduler=linear_scheduler,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            global_pruning=False,
            channel_groups=dict(),
            max_ch_sparsity=max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            customized_pruners=None,
            unwrapped_parameters=None,
            output_transform=None,
            forward_fn=None,
            root_module_types=[ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],
            out_channel_groups=dict(),
            in_channel_groups=dict(),
        )

        self.prune_weights = self._get_prune_weights()
        self.callback = self._get_admm_callback()

    def _get_prune_weights(self):        
        """Get the weights to be pruned during ADMM algorithm.

        Returns:
            dict: Dictionary containing information about the weights to be pruned.
        """
        self.current_step += 1
        prune_weights = {}

        pruning_method = self.prune_global if self.global_pruning else self.prune_local

        for group in pruning_method():
            for dep, idxs in group._group:
                if dep.target.type in [ops.OPTYPE.CONV, ops.OPTYPE.LINEAR, ops.OPTYPE.DEPTHWISE_CONV, ops.OPTYPE.PARAMETER]:
                    if dep.handler == function.PrunerBox[dep.target.type].prune_out_channels:
                        for name, param in dep.target.module.named_parameters():
                            tmp = {}
                            tmp["param_name"] = name
                            tmp["param_weight"] = param
                            tmp["dep"] = dep
                            tmp["idxs"] = idxs
                            
                            node_param_name = dep.target.name + "_" + name
                            prune_weights[node_param_name] = tmp
        self.current_step -= 1

        return prune_weights

    def _load_config(self, config):
        """Loads the configuration(json) file and returns it as a dictionary."""
        with open(config, 'r', encoding='utf-8') as f:
            return json.load(f)

    def initialize_z_u(self):
        """Initialize Z and U."""
        self.logger.info('--- ADMM Z & U Initialize ---')
        for node_param_name in self.prune_weights:
            param = self.prune_weights[node_param_name]["param_weight"]
            self.admm_u[node_param_name] = torch.zeros(param.shape, device=self.device)
            self.admm_z[node_param_name] = param.detach().clone().to(self.device)
        self._initialize_or_update_z(flag=False)

    def update_z_u(self, epoch=1, batch_idx=0):
        """Update Z and U."""
        if epoch > 0 and epoch % self.admm_epoch == 0 and batch_idx == 0:
            self._initialize_or_update_z(flag=True)
            self._update_u()
        
    def _update_u(self):
        """Update U variables."""
        for node_param_name in self.prune_weights:
            param = self.prune_weights[node_param_name]["param_weight"]
            weight = param.detach() if self.device == torch.device('cuda') else param.detach().cpu()
            u_tensor = weight - self.admm_z[node_param_name] + self.admm_u[node_param_name]
            pruner_utils.assign_tensor(self.admm_u[node_param_name], u_tensor)

    def adjust_learning_rate(self, optimizer, epoch):
        """Adjust learning rate."""
        if epoch % self.admm_epoch == 0:
            lr = self.init_lr
        else:
            offset = epoch % self.admm_epoch
            admm_step = self.admm_epoch / self.internal_stage
            lr = self.init_lr * (self.lr_decay ** (offset // admm_step))

        if isinstance(optimizer, dict):
            for key in optimizer:
                for param_group in optimizer[key].param_groups:
                    param_group['lr'] = lr
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return lr

    def get_admm_loss(self):
        """Compute the ADMM loss."""
        admm_loss = torch.tensor(0., device=self.device)
        for node_param_name in self.prune_weights:
            weight = self.prune_weights[node_param_name]["param_weight"]
            l2_norm = torch.norm(weight - self.admm_z[node_param_name] + self.admm_u[node_param_name], p=2) ** 2
            admm_loss += 0.5 * self.rho * l2_norm
        return admm_loss

    def _initialize_or_update_z(self, flag=False):
        """Initialize or update Z."""
        self.prune_weights = self._get_prune_weights()

        for node_param_name in self.prune_weights:
            param = self.prune_weights[node_param_name]["param_weight"]
            dep = self.prune_weights[node_param_name]["dep"]
            idxs = self.prune_weights[node_param_name]["idxs"]
            param_name =  self.prune_weights[node_param_name]["param_name"]

            z_next = param.detach() + self.admm_u[node_param_name] if flag else param.detach()

            updated_z = self._weight_masked(dep, idxs, param_name, z_next)
            pruner_utils.assign_tensor(self.admm_z[node_param_name], updated_z)

    def _weight_masked(self, dep, idxs, param_name, weight):
        """Apply weight masking."""
        dep.handler.__self__.pruning_dim = dep.target.pruning_dim
        if len(idxs) > 0 and isinstance(idxs[0], _helpers._HybridIndex):
            idxs = _helpers.to_plain_idxs(idxs)

        if dep.handler == function.PrunerBox[dep.target.type].prune_out_channels:
            get_weight_masked = function.PrunerBox[dep.target.type].xhqi_weight_masked_out_channels
        else:
            print("error!!!")
            import pdb;pdb.set_trace()
        weight_masked_pt = get_weight_masked(dep.target.module, weight, param_name, idxs)
        return weight_masked_pt

    def _get_admm_callback(self):
        """Get the appropriate ADMM callback."""
        if self.stage == enums.PruningStage.PRETRAIN:
            return pruner_callback.ADMMPretrainCallback(self)
        if self.stage == enums.PruningStage.PRUNE:
            return pruner_callback.ADMMPruneCallback(self)
        elif self.stage == enums.PruningStage.RETRAIN:
            return pruner_callback.ADMMRetrainCallback(self)
