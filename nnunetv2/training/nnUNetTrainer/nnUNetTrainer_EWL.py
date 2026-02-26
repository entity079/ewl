import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_EWL_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.data_augmentation.custom_transforms.Euclidean_Transform import EuclideanTransform
from nnunetv2.utilities.helpers import softmax_helper_dim1, dummy_context


class nnUNetTrainer_EWL(nnUNetTrainer):
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

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                              do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                              foreground_labels=None, regions=None, ignore_label=None):
        """
        Override to add EuclideanTransform to the training pipeline
        """
        transforms = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label
        )

        # Determine number of classes for EuclideanTransform
        if regions is not None:
            # For region-based training, num_classes = number of regions + 1 (for background)
            num_classes = len(regions) + 1
        else:
            # For regular training, num_classes = number of foreground labels + 1 (for background)
            num_classes = len(foreground_labels) + 1 if foreground_labels is not None else 1

        # Add EuclideanTransform before the final downsampling for deep supervision
        # We need to insert it before DownsampleSegForDSTransform if it exists
        euclidean_transform = EuclideanTransform(num_classes=num_classes, smooth=1.)

        # Find where to insert the EuclideanTransform
        # Insert it before DownsampleSegForDSTransform if it exists, otherwise at the end
        insert_idx = -1
        for i, transform in enumerate(transforms.transforms):
            if hasattr(transform, '__class__') and 'DownsampleSegForDSTransform' in transform.__class__.__name__:
                insert_idx = i
                break

        if insert_idx >= 0:
            transforms.transforms.insert(insert_idx, euclidean_transform)
        else:
            transforms.transforms.append(euclidean_transform)

        return transforms

    def train_step(self, batch: dict) -> dict:
        """
        Override to handle the ewtr data from EuclideanTransform
        """
        data = batch['data']
        target = batch['target']
        ewtr = batch.get('ewtr', None)  # Get the Euclidean weighted transform data

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if ewtr is not None:
            if isinstance(ewtr, list):
                ewtr = [i.to(self.device, non_blocking=True) for i in ewtr]
            else:
                ewtr = ewtr.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            if ewtr is not None:
                l = self.loss(output, target, ewtr)
            else:
                # Fallback if ewtr is not available
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
