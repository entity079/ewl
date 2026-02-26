import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, maximum, minimum

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

class EuclideanTransform(BasicTransform):
    '''This function is meant to find the Euclidean Transform of the Ground Truth Mask which is taken as the input. It converts
    the ground truth mask into a Eucildean Weighted Mask using the Euclidean Distance Transform applied on the binarized mask
    Input Ground Truth Mask Shape - [batch_size, height, width]
    '''
    def __init__(self, num_classes: int, smooth: float = 1.):
        super(EuclideanTransform, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes


    # We define the euclidean transform using the basic pytroch functions and then the loss function
    def _transform(self, binary_image: np.ndarray):
        """
        This function basically performs euclidean transform on a binary images.
        The tensor must be a 2D numpy array and must contain a single image.
        """

        # Check if the image has any background pixels (zeros)
        has_background = np.any(binary_image == 0)
        
        if has_background:
            distance_transform = distance_transform_edt(binary_image)
            
            # Check for inf values and replace them
            if np.any(np.isinf(distance_transform)):
                # If inf values exist, set them to the maximum finite distance
                finite_distances = distance_transform[np.isfinite(distance_transform)]
                if len(finite_distances) > 0:
                    max_finite = np.max(finite_distances)
                    distance_transform[np.isinf(distance_transform)] = max_finite
                else:
                    # If all are inf, set to 1
                    distance_transform = np.ones_like(distance_transform)
            
            distance_transform = (distance_transform - minimum(distance_transform)) / (maximum(distance_transform) - minimum(distance_transform))
            
            # Handle case where max == min (all same values)
            if np.isnan(distance_transform).any():
                distance_transform = np.ones_like(distance_transform) * 0.5
            
            transformed_image = 0.7 * distance_transform + 0.3 * binary_image
            
            transformed_image = (transformed_image - minimum(transformed_image)) / (maximum(transformed_image) - minimum(transformed_image))
            
            # Handle case where max == min
            if np.isnan(transformed_image).any():
                transformed_image = binary_image.astype(np.float32)
            
            return transformed_image

        # If no background pixels (entire image is foreground), return uniform weights
        else:
            return np.ones_like(binary_image, dtype=np.float32)

    # Applying the transform function to each class of a one-hot encoded tensor and its entire batch
    def apply(self, data_dict, **params):
        """
        Applies the `transform` function to each class of a one-hot encoded PyTorch tensor across the entire batch.
        Input tensor shape: [batch_size, height, width]
        """
        seg_key = 'seg' if 'seg' in data_dict else 'segmentation'
        if seg_key not in data_dict:
            raise KeyError("EuclideanTransform expected one of keys {'seg', 'segmentation'} in data_dict, "
                           f"but got keys: {list(data_dict.keys())}")

        tensor = data_dict[seg_key]
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = np.asarray(tensor)
        if tensor.ndim < 2:
            raise RuntimeError(f"EuclideanTransform expected at least 2D segmentation, got shape {tensor.shape}")

        # Transforms are applied per sample in this pipeline. Segmentation usually arrives as [C, H, W(, D)]
        # with C=1 label channel. Use channel 0 as label map to build class-wise Euclidean weights [K, H, W(, D)].
        if tensor.ndim >= 3:
            seg_map = tensor[0]
        else:
            seg_map = tensor

        spatial_shape = seg_map.shape
        image_size = np.prod(spatial_shape)
        transformed_tensor = np.zeros((self.num_classes, *spatial_shape), dtype=np.float32)

        for c in range(self.num_classes):
            bin_tensor = (seg_map == c).astype(np.float32)
            if bin_tensor.any():
                fg_sum = np.sum(bin_tensor)
                imbalance = (image_size / fg_sum)
                imbalance = min(imbalance, 100.0)  # Cap imbalance to prevent extreme values
                transformed_tensor[c] = self._transform(bin_tensor) * imbalance

        # Add nan handling
        transformed_tensor = np.nan_to_num(transformed_tensor, nan=0.0)

        data_dict["ewtr"] = torch.from_numpy(transformed_tensor)  # [num_classes, H, W(, D)]

        return data_dict