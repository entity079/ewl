import numpy as np
import torch
from torch import nn
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DC_EWL_and_CE_loss_Bounded(nn.Module):
    def __init__(self, soft_dice_kwargs, ewl_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_ew=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss,
                 bound_method='adaptive', max_ewl_ratio=2.0, clip_percentile=95):
        """
        Bounded version of DC_EWL_and_CE_loss to prevent EWL from dominating
        
        Args:
            bound_method: 'clip', 'adaptive', 'percentile', 'hybrid'
            max_ewl_ratio: Maximum ratio of EWL to base losses
            clip_percentile: Percentile for clipping method
        """
        super(DC_EWL_and_CE_loss_Bounded, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_ew = weight_ew
        self.ignore_label = ignore_label
        self.bound_method = bound_method
        self.max_ewl_ratio = max_ewl_ratio
        self.clip_percentile = clip_percentile

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        from nnunetv2.training.loss.EWL import Euclidean_Weighted_Loss
        self.ew = Euclidean_Weighted_Loss(apply_nonlin=softmax_helper_dim1, **ewl_kwargs)

    def _apply_bounds(self, ew_loss, base_loss):
        """Apply upper bounds to EWL based on selected method"""
        
        if self.bound_method == 'clip':
            # Simple clipping
            ew_loss = torch.clamp(ew_loss, max=1.0)
            
        elif self.bound_method == 'percentile':
            # Percentile-based clipping
            percentile_val = torch.quantile(ew_loss.flatten(), self.clip_percentile / 100)
            ew_loss = torch.clamp(ew_loss, max=percentile_val)
            
        elif self.bound_method == 'adaptive':
            # Adaptive scaling based on base loss magnitude
            base_mean = torch.mean(base_loss)
            ewl_mean = torch.mean(ew_loss)
            
            if ewl_mean > base_mean * self.max_ewl_ratio:
                scale_factor = (base_mean * self.max_ewl_ratio) / ewl_mean
                ew_loss = ew_loss * scale_factor
                
        elif self.bound_method == 'hybrid':
            # Hybrid: percentile + adaptive scaling
            percentile_val = torch.quantile(ew_loss.flatten(), self.clip_percentile / 100)
            ew_loss = torch.clamp(ew_loss, max=percentile_val)
            
            base_mean = torch.mean(base_loss)
            ewl_mean = torch.mean(ew_loss)
            
            if ewl_mean > base_mean * self.max_ewl_ratio:
                scale_factor = (base_mean * self.max_ewl_ratio) / ewl_mean
                ew_loss = ew_loss * scale_factor
                
        elif self.bound_method == 'sigmoid':
            # Sigmoid bounding
            temperature = 1.0
            ew_loss = torch.sigmoid(ew_loss / temperature)
            
        return ew_loss

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, ewtr: torch.Tensor):
        """
        Forward pass with bounded EWL
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            target_ew = torch.where(mask, ewtr, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_ew = ewtr
            mask = None

        # Compute individual losses
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        ew_loss = self.ew(net_output, target_ew, loss_mask=mask) \
            if self.weight_ew != 0 else 0

        # Apply bounds to EWL
        if self.weight_ew != 0:
            base_loss = self.weight_ce * ce_loss + self.weight_dice * dc_loss
            ew_loss = self._apply_bounds(ew_loss, base_loss)

        # Combine losses
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_ew * ew_loss
        return result

    def get_loss_components(self, net_output: torch.Tensor, target: torch.Tensor, ewtr: torch.Tensor):
        """
        Return individual loss components for monitoring
        """
        if self.ignore_label is not None:
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            target_ew = torch.where(mask, ewtr, 0)
        else:
            target_dice = target
            target_ew = ewtr
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) if self.weight_ce != 0 else 0
        ew_loss = self.ew(net_output, target_ew, loss_mask=mask) if self.weight_ew != 0 else 0

        return {
            'dice_loss': dc_loss.item() if hasattr(dc_loss, 'item') else dc_loss,
            'ce_loss': ce_loss.item() if hasattr(ce_loss, 'item') else ce_loss,
            'ewl_loss': ew_loss.item() if hasattr(ew_loss, 'item') else ew_loss,
            'total_loss': (self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_ew * ew_loss).item() if hasattr(self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_ew * ew_loss, 'item') else self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_ew * ew_loss
        }
