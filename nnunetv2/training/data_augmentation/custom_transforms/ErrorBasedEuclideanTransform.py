import numpy as np
import torch
import os
from scipy.ndimage import distance_transform_edt, maximum, minimum
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class ErrorBasedEuclideanTransform(BasicTransform):
    '''Error-based Euclidean Transform that computes weights based on prediction errors.
    
    Process:
    1. Subtract prediction mask from ground truth mask
    2. Filter to keep only positive values (false negatives)
    3. Apply Euclidean distance transform to false negatives
    4. Generate weighted loss focusing on missed regions
    '''
    def __init__(self, num_classes: int, smooth: float = 1., cache_dir: str = None):
        super(ErrorBasedEuclideanTransform, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.cache_dir = cache_dir
        self._cache = {}  # In-memory cache for faster access

    def _transform(self, error_mask: np.ndarray):
        """
        Apply Euclidean distance transform to error mask (false negatives).
        The tensor must be a 2D numpy array and must contain a single image.
        """
        # Check if there are any errors (false negatives)
        if error_mask.any():
            distance_transform = distance_transform_edt(error_mask)
        
            # Normalize the distance transform
            if distance_transform.max() > distance_transform.min():
                distance_transform = (distance_transform - minimum(distance_transform)) / (maximum(distance_transform) - minimum(distance_transform))
            else:
                distance_transform = np.zeros_like(distance_transform)
            
            # Apply weighting: higher weights for regions further from predicted boundaries
            transformed_image = 0.7*distance_transform + 0.3*error_mask
            
            # Final normalization
            if transformed_image.max() > transformed_image.min():
                transformed_image = (transformed_image - minimum(transformed_image)) / (maximum(transformed_image) - minimum(transformed_image))

            return transformed_image
        else:
            return error_mask

    def _compute_error_mask(self, ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """
        Compute error mask by subtracting prediction from ground truth.
        Positive values indicate false negatives (ground truth = 1, prediction = 0).
        """
        # Ensure binary masks
        gt_binary = (ground_truth > 0).astype(np.float32)
        pred_binary = (prediction > 0).astype(np.float32)
        
        # False negatives: ground truth = 1, prediction = 0
        error_mask = gt_binary - pred_binary
        
        # Keep only positive values (false negatives)
        error_mask = np.maximum(error_mask, 0)
        
        return error_mask

    def _get_cache_key(self, gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> str:
        """Generate a unique cache key based on ground truth and prediction content"""
        # Use combined hash of ground truth and prediction tensors
        gt_bytes = gt_tensor.numpy().tobytes()
        pred_bytes = pred_tensor.numpy().tobytes()
        combined_bytes = gt_bytes + pred_bytes
        return str(hash(combined_bytes))

    def _compute_error_based_euclidean_transform(self, gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> torch.Tensor:
        """Compute error-based Euclidean transform for a single sample"""
        gt_shape = gt_tensor.shape
        pred_shape = pred_tensor.shape
        image_size = np.prod(gt_shape)
        transformed_tensor = np.zeros((self.num_classes,) + gt_shape, dtype=np.float32)

        for c in range(self.num_classes):
            gt_class = (gt_tensor == c).astype(np.float32)
            pred_class = (pred_tensor == c).astype(np.float32)
            
            # Compute error mask for this class
            error_mask = self._compute_error_mask(gt_class, pred_class)
            
            if error_mask.any():
                # Apply imbalance weighting
                sum = np.sum(gt_class)
                imbalance = (image_size / sum) if sum > 0 else 1.0
                
                # Apply Euclidean transform to error mask
                transformed_class = self._transform(error_mask) * imbalance
                transformed_tensor[c] = transformed_class

        return torch.from_numpy(transformed_tensor)

    def apply(self, data_dict, **params):
        """
        Applies the error-based Euclidean transform to each sample in the batch.
        Requires both 'segmentation' (ground truth) and 'prediction' keys in data_dict.
        """
        if 'prediction' not in data_dict:
            raise ValueError("ErrorBasedEuclideanTransform requires 'prediction' key in data_dict")
            
        gt = data_dict['segmentation']
        pred = data_dict['prediction']
        batch_size = gt.shape[0]
        transformed_batch = []

        for b in range(batch_size):
            gt_tensor = gt[b]
            pred_tensor = pred[b]
            cache_key = self._get_cache_key(gt_tensor, pred_tensor)
            
            # Check in-memory cache first
            if cache_key in self._cache:
                transformed_tensor = self._cache[cache_key]
            else:
                # Compute and cache the transform
                transformed_tensor = self._compute_error_based_euclidean_transform(gt_tensor, pred_tensor)
                self._cache[cache_key] = transformed_tensor
                
                # Optionally save to disk cache
                if self.cache_dir is not None:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    cache_file = os.path.join(self.cache_dir, f"{cache_key}.pt")
                    if not os.path.exists(cache_file):
                        torch.save(transformed_tensor, cache_file)

            transformed_batch.append(transformed_tensor)

        # Stack batch dimension
        data_dict["ewtr"] = torch.stack(transformed_batch, dim=0)
        return data_dict

    def load_cache_from_disk(self):
        """Load cached transforms from disk if cache_dir is specified"""
        if self.cache_dir is not None and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pt'):
                    cache_key = filename[:-3]  # Remove .pt extension
                    cache_file = os.path.join(self.cache_dir, filename)
                    self._cache[cache_key] = torch.load(cache_file)
            print(f"Loaded {len(self._cache)} cached error-based Euclidean transforms from {self.cache_dir}")

    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache.clear()
        print("Cleared error-based Euclidean transform cache")
