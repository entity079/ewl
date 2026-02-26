import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch import autocast
from PIL import Image

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_EWL_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.data_augmentation.custom_transforms.ErrorBasedEuclideanTransform import ErrorBasedEuclideanTransform
from nnunetv2.utilities.helpers import softmax_helper_dim1, dummy_context


class nnUNetTrainer_ErrorBasedEWL_Visual(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Create cache directory for error-based Euclidean transforms
        self.error_cache_dir = os.path.join(
            self.output_folder_base, 'error_based_cache'
        ) if self.output_folder_base else None
        
        # Create visualization directory for final epoch masks
        self.visualization_dir = os.path.join(
            self.output_folder_base, 'final_epoch_visualization'
        ) if self.output_folder_base else None
        
        self._error_transform = None
        self._is_final_epoch = False
        self._saved_samples = 0
        self._max_samples_to_save = 50  # Limit number of samples to save

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

    def _save_mask_comparison(self, data, target, prediction, error_weights, batch_idx):
        """
        Save comparison of ground truth, prediction, and error weights
        """
        if self.visualization_dir is None or self._saved_samples >= self._max_samples_to_save:
            return

        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Get first sample from batch (assuming 2D data)
        if len(data.shape) == 4:  # [B, C, H, W]
            data_sample = data[0].cpu().numpy()
            target_sample = target[0].cpu().numpy() if len(target.shape) == 4 else target[0][0].cpu().numpy()
            pred_sample = prediction[0].cpu().numpy()
            error_sample = error_weights[0].cpu().numpy() if error_weights is not None else None
        else:
            return  # Skip if not 2D data

        # Handle different target formats
        if len(target_sample.shape) == 3:  # [C, H, W]
            target_2d = np.argmax(target_sample, axis=0)
        else:  # [H, W]
            target_2d = target_sample

        # Handle prediction format
        if len(pred_sample.shape) == 3:  # [C, H, W]
            pred_2d = np.argmax(pred_sample, axis=0)
        else:  # [H, W]
            pred_2d = pred_sample

        # Handle input data (use first channel if multi-channel)
        if len(data_sample.shape) == 3:  # [C, H, W]
            input_2d = data_sample[0]
        else:  # [H, W]
            input_2d = data_sample

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Final Epoch - Sample {self._saved_samples + 1} (Batch {batch_idx})', fontsize=16)

        # Input image
        axes[0, 0].imshow(input_2d, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        # Ground truth
        axes[0, 1].imshow(target_2d, cmap='jet')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')

        # Prediction
        axes[1, 0].imshow(pred_2d, cmap='jet')
        axes[1, 0].set_title('Model Prediction')
        axes[1, 0].axis('off')

        # Error weights or difference
        if error_sample is not None and len(error_sample.shape) == 3:
            # Sum error weights across classes for visualization
            error_2d = np.sum(error_sample, axis=0)
            axes[1, 1].imshow(error_2d, cmap='hot')
            axes[1, 1].set_title('Error Weights')
        else:
            # Show difference between prediction and ground truth
            diff = np.abs(pred_2d.astype(float) - target_2d.astype(float))
            axes[1, 1].imshow(diff, cmap='Reds')
            axes[1, 1].set_title('Prediction Error')
        axes[1, 1].axis('off')

        # Save the figure
        save_path = os.path.join(self.visualization_dir, f'comparison_sample_{self._saved_samples + 1:03d}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Also save individual masks as numpy arrays for further analysis
        np.save(os.path.join(self.visualization_dir, f'input_{self._saved_samples + 1:03d}.npy'), input_2d)
        np.save(os.path.join(self.visualization_dir, f'ground_truth_{self._saved_samples + 1:03d}.npy'), target_2d)
        np.save(os.path.join(self.visualization_dir, f'prediction_{self._saved_samples + 1:03d}.npy'), pred_2d)
        if error_sample is not None:
            np.save(os.path.join(self.visualization_dir, f'error_weights_{self._saved_samples + 1:03d}.npy'), error_sample)

        self._saved_samples += 1
        self.print_to_log_file(f"Saved mask comparison for sample {self._saved_samples}/{self._max_samples_to_save}")

    def train_step(self, batch: dict) -> dict:
        """
        Override to handle error-based Euclidean transforms and save masks in final epoch
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

            # Save mask comparisons in final epoch
            if self._is_final_epoch and self.current_epoch % 10 == 0:  # Save every 10 batches in final epoch
                self._save_mask_comparison(
                    data=data,
                    target=target,
                    prediction=pred_classes,
                    error_weights=ewtr,
                    batch_idx=self.current_batch_index
                )

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

    def on_epoch_start(self):
        """Override to detect final epoch"""
        super().on_epoch_start()
        
        # Check if this is the final epoch
        if hasattr(self, 'num_epochs') and self.current_epoch >= self.num_epochs - 1:
            if not self._is_final_epoch:
                self._is_final_epoch = True
                self.print_to_log_file(f"FINAL EPOCH DETECTED: Starting mask visualization saving")
                if self.visualization_dir:
                    self.print_to_log_file(f"Visualizations will be saved to: {self.visualization_dir}")

    def on_train_start(self):
        """Override to initialize cache before training"""
        super().on_train_start()
        
        # Print cache information
        if self.error_cache_dir:
            self.print_to_log_file(f"Error-based Euclidean transform cache directory: {self.error_cache_dir}")
        
        if self.visualization_dir:
            self.print_to_log_file(f"Final epoch visualization directory: {self.visualization_dir}")
        
        if self._error_transform:
            cache_size = len(self._error_transform._cache)
            self.print_to_log_file(f"Initial cache size: {cache_size} transforms")

    def on_train_end(self):
        """Override to save cache statistics and visualization summary"""
        super().on_train_end()
        
        if self._error_transform:
            cache_size = len(self._error_transform._cache)
            self.print_to_log_file(f"Final cache size: {cache_size} transforms")
            
            if self.error_cache_dir:
                cache_files = len([f for f in os.listdir(self.error_cache_dir) if f.endswith('.pt')]) \
                    if os.path.exists(self.error_cache_dir) else 0
                self.print_to_log_file(f"Cache files saved: {cache_files}")

        # Visualization summary
        if self.visualization_dir and os.path.exists(self.visualization_dir):
            vis_files = len([f for f in os.listdir(self.visualization_dir) if f.endswith('.png')])
            mask_files = len([f for f in os.listdir(self.visualization_dir) if f.endswith('.npy')])
            self.print_to_log_file(f"Final epoch visualizations saved: {vis_files} PNG files")
            self.print_to_log_file(f"Mask arrays saved: {mask_files} NPY files")
            self.print_to_log_file(f"Check {self.visualization_dir} for detailed mask comparisons")

    def clear_error_cache(self):
        """Method to manually clear the error-based cache if needed"""
        if self._error_transform:
            self._error_transform.clear_cache()
            self.print_to_log_file("Error-based Euclidean transform cache cleared manually")
