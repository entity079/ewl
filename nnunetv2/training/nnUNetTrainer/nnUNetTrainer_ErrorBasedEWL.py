import numpy as np
import torch
import os
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_EWL_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.data_augmentation.custom_transforms.ErrorBasedEuclideanTransform import ErrorBasedEuclideanTransform
from nnunetv2.utilities.helpers import softmax_helper_dim1, dummy_context


class nnUNetTrainer_ErrorBasedEWL(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Create cache directory for error-based Euclidean transforms
        self.error_cache_dir = os.path.join(
            self.output_folder_base, 'error_based_cache'
        ) if self.output_folder_base else None
        
        self._error_transform = None

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_EWL_and_CE_loss(
                soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': True, 'ddp': self.is_ddp},
                ewl_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                           'smooth': 1e-5, 'ddp': self.is_ddp},
                ce_kwargs={}, weight_ce=1, weight_dice=1, weight_ew=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            loss = DC_EWL_and_CE_loss(
                soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ewl_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                           'smooth': 1e-5, 'ddp': self.is_ddp},
                ce_kwargs={}, weight_ce=1, weight_dice=1, weight_ew=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                              do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                              foreground_labels=None, regions=None, ignore_label=None):
        """
        Override to add ErrorBasedEuclideanTransform to the training pipeline
        """
        transforms = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label
        )

        # Determine number of classes for ErrorBasedEuclideanTransform
        if regions is not None:
            num_classes = len(regions) + 1
        else:
            num_classes = len(foreground_labels) + 1 if foreground_labels is not None else 1

        # Create error-based Euclidean transform
        self._error_transform = ErrorBasedEuclideanTransform(
            num_classes=num_classes, 
            smooth=1., 
            cache_dir=self.error_cache_dir
        )
        
        # Load existing cache from disk
        self._error_transform.load_cache_from_disk()

        # Insert ErrorBasedEuclideanTransform before DownsampleSegForDSTransform if it exists
        insert_idx = -1
        for i, transform in enumerate(transforms.transforms):
            if hasattr(transform, '__class__') and 'DownsampleSegForDSTransform' in transform.__class__.__name__:
                insert_idx = i
                break

        if insert_idx >= 0:
            transforms.transforms.insert(insert_idx, self._error_transform)
        else:
            transforms.transforms.append(self._error_transform)

        return transforms

    def train_step(self, batch: dict) -> dict:
        """
        Override to handle error-based Euclidean transforms
        """
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            
            # Generate predictions for error-based transform
            # Convert network output to class predictions (argmax)
            if isinstance(output, list):
                # Deep supervision case - use highest resolution output
                pred_classes = torch.argmax(output[0], dim=1, keepdim=True)
            else:
                pred_classes = torch.argmax(output, dim=1, keepdim=True)
            
            # Add prediction to batch for error-based transform
            batch_with_pred = {
                'data': data,
                'target': target,
                'segmentation': target[0] if isinstance(target, list) else target,
                'prediction': pred_classes
            }
            
            # Apply error-based transform
            if self._error_transform:
                batch_with_pred = self._error_transform.apply(batch_with_pred)
                ewtr = batch_with_pred.get('ewtr', None)
            else:
                ewtr = None

            # Compute loss with error-based weights
            if ewtr is not None:
                if isinstance(ewtr, list):
                    ewtr = [i.to(self.device, non_blocking=True) for i in ewtr]
                else:
                    ewtr = ewtr.to(self.device, non_blocking=True)
                l = self.loss(output, target, ewtr)
            else:
                # Fallback if error-based transform is not available
                l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def on_train_start(self):
        """Override to initialize cache before training"""
        super().on_train_start()
        
        # Print cache information
        if self.error_cache_dir:
            self.print_to_log_file(f"Error-based Euclidean transform cache directory: {self.error_cache_dir}")
        
        if self._error_transform:
            cache_size = len(self._error_transform._cache)
            self.print_to_log_file(f"Initial cache size: {cache_size} transforms")

    def on_train_end(self):
        """Override to save cache statistics"""
        super().on_train_end()
        
        if self._error_transform:
            cache_size = len(self._error_transform._cache)
            self.print_to_log_file(f"Final cache size: {cache_size} transforms")
            
            if self.error_cache_dir:
                cache_files = len([f for f in os.listdir(self.error_cache_dir) if f.endswith('.pt')]) \
                    if os.path.exists(self.error_cache_dir) else 0
                self.print_to_log_file(f"Cache files saved: {cache_files}")

    def clear_error_cache(self):
        """Method to manually clear the error-based cache if needed"""
        if self._error_transform:
            self._error_transform.clear_cache()
            self.print_to_log_file("Error-based Euclidean transform cache cleared manually")
