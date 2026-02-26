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
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.data_augmentation.custom_transforms.Euclidean_Transform import EuclideanTransform
from nnunetv2.utilities.helpers import softmax_helper_dim1, dummy_context


class nnUNetTrainer_EWL_Unified(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Create cache directory for precomputed Euclidean transforms
        self.euclidean_cache_dir = os.path.join(
            self.output_folder_base, 'euclidean_cache'
        ) if self.output_folder_base else None
        
        # Create visualization directory for final epoch masks
        self.visualization_dir = os.path.join(
            self.output_folder_base, 'final_epoch_visualization'
        ) if self.output_folder_base else None
        
        # Initialize Euclidean transform object
        self._euclidean_transform = None
        # Flag to track if this is the final epoch (for visualization)
        self._is_final_epoch = False
        # Counter for saved visualization samples
        self._saved_samples = 0
        # Limit number of visualization samples to save (prevents storage overflow)
        self._max_samples_to_save = 10  
        # Flag to track if Euclidean masks have been precomputed (avoids redundant computation)
        self._euclidean_masks_precomputed = False
        self._euclidean_masks_saved = 0

        # Save some visualizations early so files are visible immediately during training.
        self._save_visualizations_every_n_batches = 10
        self._global_train_batch_counter = 0

        # EWL is numerically sensitive with AMP + compile on some stacks. Use safer defaults.
        self.initial_lr = 1e-3
        self.use_amp = False
        self.grad_scaler = None

    def _do_i_compile(self):
        # torch.compile has shown unstable behavior with this custom trainer/loss stack.
        return False

    def _build_loss(self):
        """
        Build loss with adaptive scaling (your preferred method)
        """
        # Check if dataset uses region-based labels (e.g., left/right organs grouped)
        if self.label_manager.has_regions:
            # For region-based datasets: use background-aware Dice loss
            loss = DC_EWL_and_CE_loss(
                # Soft Dice loss parameters - includes background class for regions
                soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': True, 'ddp': self.is_ddp},
                # Euclidean Weighted Loss parameters - same as Dice for consistency
                ewl_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                           'smooth': 1e-5, 'ddp': self.is_ddp},
                # Cross-entropy loss parameters (empty dict = default settings)
                ce_kwargs={}, 
                # Loss component weights: equal weighting for CE, Dice, and EWL
                weight_ce=1, weight_dice=1, weight_ew=1,
                # Ignore label for unlabeled regions (from dataset configuration)
                ignore_label=self.label_manager.ignore_label,
                # Use memory-efficient Dice loss implementation
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            # For standard class-based datasets: exclude background from Dice loss
            loss = DC_EWL_and_CE_loss(
                # Soft Dice loss parameters - excludes background class for standard classes
                soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                # Euclidean Weighted Loss parameters - same as Dice for consistency
                ewl_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                           'smooth': 1e-5, 'ddp': self.is_ddp},
                # Cross-entropy loss parameters (empty dict = default settings)
                ce_kwargs={}, 
                # Loss component weights: equal weighting for CE, Dice, and EWL
                weight_ce=1, weight_dice=1, weight_ew=1,
                # Ignore label for unlabeled regions (from dataset configuration)
                ignore_label=self.label_manager.ignore_label,
                # Use memory-efficient Dice loss implementation
                dice_class=MemoryEfficientSoftDiceLoss
            )

        # If torch.compile is enabled, compile the Dice loss for better performance
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # If deep supervision is enabled (multiple output resolutions), wrap loss
        if self.enable_deep_supervision:
            # Get the scales for deep supervision (different resolution levels)
            deep_supervision_scales = self._get_deep_supervision_scales()
            # Create weights for different supervision levels (exponential decay)
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            # Special handling for distributed training (DDP)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6  # Very small weight for highest resolution to avoid gradient conflicts
            else:
                weights[-1] = 0  # Zero weight for highest resolution (standard practice)
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _precompute_euclidean_masks(self, dataset):
        """Precompute Euclidean masks once for the entire dataset"""
        # Check if cache exists and is not empty
        if self._euclidean_masks_precomputed and self.euclidean_cache_dir and os.path.exists(self.euclidean_cache_dir):
            cache_files = [f for f in os.listdir(self.euclidean_cache_dir) if f.endswith('.npy')]
            if len(cache_files) > 0:
                return  # Cache exists, skip
        
        self.print_to_log_file("Precomputing Euclidean masks for entire dataset...")
        
        # Create cache directory
        if self.euclidean_cache_dir:
            os.makedirs(self.euclidean_cache_dir, exist_ok=True)
        
        # Determine number of classes
        if hasattr(self, 'label_manager') and hasattr(self.label_manager, 'all_labels'):
            num_classes = len(self.label_manager.all_labels)
        else:
            num_classes = 2  # Default binary case
            
        # Create Euclidean transform
        self._euclidean_transform = EuclideanTransform(num_classes=num_classes, smooth=1.)
        
        # Iterate through dataset and precompute transforms
        identifiers = list(dataset.identifiers)
        total_cases = len(identifiers)
        
        saved = 0
        for i, case_id in enumerate(identifiers):
            try:
                case_data = dataset[case_id]
                # nnUNet datasets return tuples: (data, seg, seg_prev, properties)
                if isinstance(case_data, (tuple, list)):
                    gt_mask = case_data[1]
                else:
                    gt_mask = case_data['seg']

                # Apply Euclidean transform
                data_dict = {'seg': gt_mask}
                transformed = self._euclidean_transform(**data_dict)
                ewtr = transformed['ewtr']
                if isinstance(ewtr, torch.Tensor):
                    ewtr = ewtr.detach().cpu().numpy()

                # Save to cache
                cache_file = os.path.join(self.euclidean_cache_dir, f'case_{case_id}.npy')
                np.save(cache_file, ewtr)
                saved += 1
            except Exception as e:
                self.print_to_log_file(f"Skipping case {case_id} during Euclidean precompute due to error: {e}")

            if (i + 1) % 50 == 0:  # Progress update every 50 cases
                self.print_to_log_file(f"Precomputed {i + 1}/{total_cases} cases (saved: {saved})")

        self._euclidean_masks_saved = saved
        self._euclidean_masks_precomputed = saved > 0
        self.print_to_log_file(
            f"Euclidean transform precomputed for {saved}/{total_cases} cases, {num_classes} classes")

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
            num_classes = len(regions) + 1
        else:
            num_classes = len(foreground_labels) + 1 if foreground_labels is not None else 1

        # Create Euclidean transform (precomputed approach)
        self._euclidean_transform = EuclideanTransform(num_classes=num_classes, smooth=1.)

        # Insert EuclideanTransform before DownsampleSegForDSTransform if it exists
        insert_idx = -1
        for i, transform in enumerate(transforms.transforms):
            if hasattr(transform, '__class__') and 'DownsampleSegForDSTransform' in transform.__class__.__name__:
                insert_idx = i
                break

        if insert_idx >= 0:
            transforms.transforms.insert(insert_idx, self._euclidean_transform)
        else:
            transforms.transforms.append(self._euclidean_transform)

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
            
            # Handle target which might be list (deep supervision) or tensor
            if isinstance(target, list):
                target_sample = target[0][0].cpu().numpy() if len(target[0].shape) == 4 else target[0][0].cpu().numpy()
            else:
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
            axes[1, 1].set_title('Euclidean Weights')
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
            np.save(os.path.join(self.visualization_dir, f'euclidean_weights_{self._saved_samples + 1:03d}.npy'), error_sample)

        self._saved_samples += 1
        self.print_to_log_file(f"Saved mask comparison for sample {self._saved_samples}/{self._max_samples_to_save}")

    def train_step(self, batch: dict) -> dict:
        """
        Override to handle Euclidean weights and save masks in final epoch
        """
        self._global_train_batch_counter += 1
        data = batch['data']
        target = batch['target']
        ewtr = batch.get('ewtr', None)  # Get Euclidean weighted transform data

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if ewtr is None:
            raise ValueError(
                "Batch is missing 'ewtr'. Please ensure the dataloader propagates ewtr from EuclideanTransform "
                "and do not use ones-like fallback."
            )
        if isinstance(ewtr, (list, tuple)):
            ewtr = [i.to(self.device, non_blocking=True) for i in ewtr]
            if any((isinstance(i, torch.Tensor) and i.numel() == 0) for i in ewtr):
                raise ValueError("Batch contains an empty ewtr tensor in deep supervision levels.")
        else:
            ewtr = ewtr.to(self.device, non_blocking=True)
            if isinstance(ewtr, torch.Tensor) and ewtr.numel() == 0:
                raise ValueError("Batch contains an empty ewtr tensor.")

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=self.use_amp) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            
            # Generate predictions for visualization
            if isinstance(output, list):
                pred_classes = torch.argmax(output[0], dim=1, keepdim=True)
            else:
                pred_classes = torch.argmax(output, dim=1, keepdim=True)

            # Compute loss with Euclidean weights
            # Handle deep supervision case where all arguments need to be tuples
            if isinstance(output, (list, tuple)):
                # Deep supervision is enabled - output and target are tuples
                if isinstance(ewtr, (list, tuple)):
                    ewtr_levels = list(ewtr)
                    if len(ewtr_levels) < len(output):
                        ewtr_levels += [None] * (len(output) - len(ewtr_levels))
                    ewtr_levels = tuple(ewtr_levels[:len(output)])
                    l = self.loss(output, target, ewtr_levels)
                else:
                    # Use Euclidean weights only at highest resolution.
                    # Lower-resolution DS heads receive None to avoid shape mismatches and instability.
                    ewtr_levels = (ewtr,) + (None,) * (len(output) - 1)
                    l = self.loss(output, target, ewtr_levels)
            else:
                # No deep supervision - pass arguments directly
                l = self.loss(output, target, ewtr)

            if isinstance(l, torch.Tensor):
                l = torch.nan_to_num(l, nan=0.0, posinf=1e3, neginf=-1e3)

            # Save mask comparisons in epoch 10 and final epoch to reduce IO/memory overhead
            batch_idx = self._global_train_batch_counter
            is_tenth_epoch = self.current_epoch == 9
            should_save_first = is_tenth_epoch and batch_idx % self._save_visualizations_every_n_batches == 0
            should_save_final = self._is_final_epoch and batch_idx % self._save_visualizations_every_n_batches == 0
            if should_save_first or should_save_final:
                self._save_mask_comparison(
                    data=data,
                    target=target,
                    prediction=pred_classes,
                    error_weights=ewtr,
                    batch_idx=batch_idx
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

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=self.use_amp) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            if isinstance(output, (list, tuple)):
                l = self.loss(output, target, (None,) * len(output))
            else:
                l = self.loss(output, target, None)
            l = torch.nan_to_num(l, nan=0.0, posinf=1e3, neginf=-1e3)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

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

    def initialize(self):
        """Override to setup dataloaders and precompute Euclidean masks"""
        super().initialize()
        
        # Precomputation moved to on_train_start after dataloaders are created

    def on_train_start(self):
        """Override to initialize cache and precompute Euclidean masks after dataloaders"""
        super().on_train_start()
        
        # Precompute Euclidean masks after dataloaders are created
        if not self._euclidean_masks_precomputed:
            # Create dataset directly using the dataset class
            try:
                dataset = self.dataset_class(
                    self.preprocessed_dataset_folder,
                    identifiers=None,  # Get all cases
                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
                )
                self._precompute_euclidean_masks(dataset)
            except Exception as e:
                self.print_to_log_file(f"Warning: Could not create dataset for precomputation: {e}")
        
        # Print directory information
        if self.euclidean_cache_dir:
            self.print_to_log_file(f"Euclidean cache directory: {self.euclidean_cache_dir}")
        
        if self.visualization_dir:
            self.print_to_log_file(f"Final epoch visualization directory: {self.visualization_dir}")
            self.print_to_log_file(f"Check {self.visualization_dir} for detailed mask comparisons")

        if self.euclidean_cache_dir and os.path.exists(self.euclidean_cache_dir):
            cache_files = len([f for f in os.listdir(self.euclidean_cache_dir) if f.endswith('.npy')]) \
                if os.path.exists(self.euclidean_cache_dir) else 0
            self.print_to_log_file(f"Euclidean cache files: {cache_files}")
