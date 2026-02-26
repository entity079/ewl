import numpy as np
import torch
import os
from scipy.ndimage import distance_transform_edt, maximum, minimum
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class CachedEuclideanTransform(BasicTransform):
    '''Optimized EuclideanTransform that precomputes and caches ground truth transforms.
    Since ground truth doesn't change during training, we compute once and reuse.
    '''
    def __init__(self, num_classes: int, smooth: float = 1., cache_dir: str = None):
        super(CachedEuclideanTransform, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.cache_dir = cache_dir
        self._cache = {}  # In-memory cache for faster access

    def _transform(self, binary_image: np.ndarray):
        """
        This function basically performs euclidean transform on a binary images.
        The tensor must be a 2D numpy array and must contain a single image.
        """
        # Check does the image even contain something or just an empty background image
        if binary_image.any():
            distance_transform = distance_transform_edt(binary_image)
        
            distance_transform = (distance_transform - minimum(distance_transform)) / (maximum(distance_transform) - minimum(distance_transform))
            
            transformed_image = 0.7*distance_transform + 0.3*binary_image
            transformed_image = (transformed_image - minimum(transformed_image)) / (maximum(transformed_image) - minimum(transformed_image))

            return transformed_image
        else:
            return binary_image

    def _get_cache_key(self, seg_tensor: torch.Tensor) -> str:
        """Generate a unique cache key based on segmentation content"""
        # Use hash of segmentation tensor as key
        seg_bytes = seg_tensor.numpy().tobytes()
        return str(hash(seg_bytes))

    def _compute_euclidean_transform(self, seg_tensor: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean transform for a single segmentation tensor"""
        tensor = seg_tensor.numpy()
        shape = tensor.shape
        image_size = np.prod(shape)
        transformed_tensor = np.zeros((self.num_classes,) + shape, dtype=np.float32)

        for c in range(self.num_classes):
            bin_tensor = (tensor == c).astype(np.float32)
            if bin_tensor.any():
                sum = np.sum(bin_tensor)
                imbalance = (image_size / sum)
                transformed_tensor[c] = self._transform(bin_tensor) * imbalance

        return torch.from_numpy(transformed_tensor)

    def apply(self, data_dict, **params):
        """
        Applies the cached Euclidean transform to each sample in the batch.
        """
        seg = data_dict['segmentation']
        batch_size = seg.shape[0]
        transformed_batch = []

        for b in range(batch_size):
            seg_tensor = seg[b]
            cache_key = self._get_cache_key(seg_tensor)
            
            # Check in-memory cache first
            if cache_key in self._cache:
                transformed_tensor = self._cache[cache_key]
            else:
                # Compute and cache the transform
                transformed_tensor = self._compute_euclidean_transform(seg_tensor)
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
            print(f"Loaded {len(self._cache)} cached Euclidean transforms from {self.cache_dir}")

    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache.clear()
        print("Cleared Euclidean transform cache")
