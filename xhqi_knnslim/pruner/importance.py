import abc
import torch
import torch.nn as nn

import typing
from . import function
from ..dependency import Group
from .._helpers import _FlattenIndexMapping
from .. import ops
import math


class Importance(abc.ABC):
    """ Estimate the importance of a tp.Dependency.Group, and return an 1-D per-channel importance score.

        It should accept a group and a ch_groups as inputs, and return a 1-D tensor with the same length as the number of channels.
        ch_groups refer to the number of internal groups, e.g., for a 64-channel **group conv** with groups=ch_groups=4, each group has 16 channels.
        All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
        Just ignore the ch_groups if you are not familar with grouping.

        Example:
            ```python
            DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
            group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
            scorer = MagnitudeImportance()    
            imp_score = scorer(group, ch_groups=1)    
            #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
            min_score = imp_score.min() 
            ``` 
    """
    @abc.abstractclassmethod
    def __call__(self, group: Group, ch_groups: int=1) -> torch.Tensor: 
        raise NotImplementedError


class MagnitudeImportance(Importance):
    """ A general implementation of magnitude importance. By default, it calculates the group L2-norm for each channel/dim.
        MagnitudeImportance supports several variants:
            - Standard L1-norm for single layer: MagnitudeImportance(p=1, normalizer=None, group_reduction="first")
            - Group L1-Norm: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean")
            - BN Scaling Factor: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean", target_types=[nn.modules.batchnorm._BatchNorm])

        Args:
            * p (int): the norm degree. Default: 2
            * group_reduction (str): the reduction method for group importance. Default: "mean"
            * normalizer (str): the normalization method for group importance. Default: "mean"
            * target_types (list): the target types for importance calculation. Default: [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]
    """
    def __init__(self, p: int=2, group_reduction: str="mean", normalizer: str='mean', target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types

    def _normalize(self, group_importance, normalizer):
        if normalizer is None:
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        else:
            raise NotImplementedError

    def _reduce(self, group_imp: typing.List[torch.Tensor], group_idxs: typing.List[typing.List[int]]):
        if len(group_imp) == 0: return group_imp
        if self.group_reduction == 'prod':
            reduced_imp = torch.ones_like(group_imp[0])
        elif self.group_reduction == 'max':
            reduced_imp = torch.ones_like(group_imp[0]) * -99999
        else:
            reduced_imp = torch.zeros_like(group_imp[0])

        for i, (imp, root_idxs) in enumerate(zip(group_imp, group_idxs)):
            if self.group_reduction == "sum" or self.group_reduction == "mean":
                reduced_imp.scatter_add_(0, torch.tensor(root_idxs, device=imp.device), imp) # accumulated importance
            elif self.group_reduction == "max": # keep the max importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                selected_imp = torch.maximum(input=selected_imp, other=imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == "prod": # product of importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                torch.mul(selected_imp, imp, out=selected_imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == 'first':
                if i == 0:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction == 'gate':
                if i == len(group_imp)-1:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction is None:
                reduced_imp = torch.stack(group_imp, dim=0) # no reduction
            else:
                raise NotImplementedError
        
        if self.group_reduction == "mean":
            reduced_imp /= len(group_imp)
        return reduced_imp
        
    @torch.no_grad()
    def __call__(self, group: Group, ch_groups: int=1):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)

                local_imp = w.abs().pow(self.p).sum(1)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(ch_groups)
                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            ####################
            # BatchNorm
            ####################
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class GroupNormImportance(MagnitudeImportance):
    """ DepGraph: Towards Any Structural Pruning. 
    https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
    """
    pass


class BNScaleImportance(MagnitudeImportance):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519
    """

    def __init__(self, group_reduction='mean', normalizer='mean'):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer, target_types=(nn.modules.batchnorm._BatchNorm,))


class LAMPImportance(MagnitudeImportance):
    """Layer-adaptive Sparsity for the Magnitude-based Pruning,
    https://arxiv.org/abs/2010.07611
    """

    def __init__(self, p=2, group_reduction="mean", normalizer='mean'):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer)

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = super().__call__(group, ch_groups)
        return self.lamp(group_imp)

    def lamp(self, imp):
        argsort_idx = torch.argsort(imp, dim=0, descending=True).tolist()
        sorted_imp = imp[argsort_idx]
        cumsum_imp = torch.cumsum(sorted_imp, dim=0)
        sorted_imp = sorted_imp / cumsum_imp
        inversed_idx = torch.arange(len(sorted_imp))[
            argsort_idx
        ].tolist()  # [0, 1, 2, 3, ..., ]
        return sorted_imp[inversed_idx]


class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        return torch.rand(len(idxs))


class TaylorImportance(MagnitudeImportance):
    """First-order taylor expansion of the loss function.
       https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf
    """
    def __init__(self, group_reduction:str="mean", normalizer:str='mean', multivariable:bool=False, target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    dw = layer.weight.grad.data.transpose(1, 0)[
                        idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                    dw = layer.weight.grad.data[idxs].flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)[idxs]
                    dw = (layer.weight.grad).flatten(1)[idxs]
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)[idxs]
                    dw = (layer.weight.grad).transpose(0, 1).flatten(1)[idxs]
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp
