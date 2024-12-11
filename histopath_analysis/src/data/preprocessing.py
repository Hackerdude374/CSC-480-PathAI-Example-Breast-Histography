import numpy as np
import torch
from PIL import Image
from typing import List, Tuple
from torch_geometric.data import Data
import cv2
from skimage.feature import peak_local_max
from scipy.spatial import Delaunay

# def extract_patches(
#     image: Image.Image,
#     patch_size: int = 50,
#     num_patches: int = 100,
#     overlap: float = 0.5
# ) -> List[Image.Image]:
#     """Extract patches from whole slide image"""
#     width, height = image.size
#     stride = int(patch_size * (1 - overlap))
    
#     # Get all possible patch coordinates
#     x_coords = range(0, width - patch_size + 1, stride)
#     y_coords = range(0, height - patch_size + 1, stride)
    
#     patches = []
#     for x in x_coords:
#         for y in y_coords:
#             patch = image.crop((x, y, x + patch_size, y + patch_size))
#             if _is_tissue_patch(patch):
#                 patches.append(patch)
    
#     # Random sample if we have more patches than needed
#     if len(patches) > num_patches:
#         indices = np.random.choice(
#             len(patches),
#             num_patches,
#             replace=False
#         )
#         patches = [patches[i] for i in indices]
    
#     # Pad with empty patches if we have too few
#     while len(patches) < num_patches:
#         patches.append(Image.fromarray(np.zeros((patch_size, patch_size, 3))))
    
#     return patches
def extract_patches(
    image: Image.Image,
    patch_size: int = 50,
    num_patches: int = 100,
    overlap: float = 0.5
) -> List[Image.Image]:
    """Extract patches from whole slide image"""
    width, height = image.size
    stride = int(patch_size * (1 - overlap))
    
    # Get all possible patch coordinates
    x_coords = range(0, width - patch_size + 1, stride)
    y_coords = range(0, height - patch_size + 1, stride)
    
    patches = []
    for x in x_coords:
        for y in y_coords:
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            if _is_tissue_patch(patch):
                patches.append(patch)
    
    # Random sample if we have more patches than needed
    if len(patches) > num_patches:
        indices = np.random.choice(len(patches), num_patches, replace=False)
        patches = [patches[i] for i in indices]
    
    # Pad with empty patches if we have too few
    while len(patches) < num_patches:
        # Create empty patch with correct data type
        empty_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)  # Change to uint8
        patches.append(Image.fromarray(empty_patch))
    
    return patches


def _is_tissue_patch(patch: Image.Image, threshold: float = 0.1) -> bool:
    """Check if patch contains tissue (not background)"""
    gray = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2GRAY)
    return np.mean(gray < 220) > threshold


def create_tissue_graph(patches: torch.Tensor) -> Data:
    """Create a graph from tissue patches using nuclei detection"""
    # Convert patches to numpy for OpenCV processing
    patches_np = patches.permute(0, 2, 3, 1).numpy()
    
    # Initialize lists for storing features and positions
    node_features = []
    node_positions = []
    
    for patch in patches_np:
        # Convert to grayscale
        gray = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Detect nuclei using local maxima
        nuclei_coords = peak_local_max(
            -gray,
            min_distance=10,
            threshold_abs=50,
            exclude_border=False
        )
        
        # Extract features for each nucleus
        for coord in nuclei_coords:
            y, x = coord
            # Get patch features around nucleus
            feature_patch = patch[
                max(0, y-5):min(patch.shape[0], y+5),
                max(0, x-5):min(patch.shape[1], x+5)
            ]
            # Use mean color and intensity as features
            features = np.concatenate([
                np.mean(feature_patch, axis=(0, 1)),
                np.array([gray[y, x] / 255.0])
            ])
            node_features.append(features)
            node_positions.append([x, y])

    # If no nodes were detected, create a dummy node
    if len(node_features) == 0:
        node_features = [[0.0] * 4]  # 3 color channels + 1 intensity
        node_positions = [[0.0, 0.0]]

    # Convert lists to numpy arrays first, then to tensors
    node_features = np.array(node_features, dtype=np.float32)
    node_positions = np.array(node_positions, dtype=np.float32)
    
    # Convert to tensors
    node_features = torch.from_numpy(node_features)
    node_positions = torch.from_numpy(node_positions)
    
    # Create edges using Delaunay triangulation
    if len(node_positions) > 3:
        tri = Delaunay(node_positions.numpy())
        edges = []
        for simplex in tri.simplices:
            for i in range(3):
                edges.append([simplex[i], simplex[(i + 1) % 3]])
                edges.append([simplex[(i + 1) % 3], simplex[i]])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    else:
        # Fallback for too few nodes: create fully connected graph
        num_nodes = len(node_positions)
        edges = [
            [i, j] for i in range(num_nodes) for j in range(num_nodes)
            if i != j
        ]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Create PyG Data object
    graph = Data(
        x=node_features,
        edge_index=edge_index,
        pos=node_positions
    )
    
    return graph

def normalize_staining(
    image: np.ndarray,
    target_means: Tuple[float, float, float] = (0.8, 0.8, 0.8),
    target_stds: Tuple[float, float, float] = (0.1, 0.1, 0.1)
) -> np.ndarray:
    """Normalize H&E staining colors"""
    # Convert to optical density
    od = -np.log((image.astype(float) + 1) / 256)
    
    # Calculate current means and stds
    means = np.mean(od, axis=(0, 1))
    stds = np.std(od, axis=(0, 1))
    
    # Normalize
    normalized = np.zeros_like(od)
    for i in range(3):
        normalized[:, :, i] = ((od[:, :, i] - means[i]) 
                             * (target_stds[i] / stds[i]) 
                             + target_means[i])
    
    # Convert back to RGB
    normalized = np.exp(-normalized) * 256 - 1
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized

def augment_patch(
    patch: Image.Image,
    augmentation_params: dict
) -> Image.Image:
    """Apply augmentations to a patch"""
    # Convert to numpy array
    patch_np = np.array(patch)
    
    if augmentation_params.get('flip_horizontal', False):
        if np.random.rand() > 0.5:
            patch_np = np.fliplr(patch_np)
            
    if augmentation_params.get('flip_vertical', False):
        if np.random.rand() > 0.5:
            patch_np = np.flipud(patch_np)
            
    if augmentation_params.get('rotate', False):
        angle = np.random.randint(0, 360)
        patch_np = rotate_image(patch_np, angle)
        
    if augmentation_params.get('color_jitter', False):
        patch_np = apply_color_jitter(patch_np)
        
    return Image.fromarray(patch_np)

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT
    )
    
    return rotated

def apply_color_jitter(
    image: np.ndarray,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.1
) -> np.ndarray:
    """Apply color jittering to image"""
    # Convert to HSV for easier manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Random adjustments
    hsv[:, :, 0] += np.random.uniform(-hue, hue) * 180  # Hue
    hsv[:, :, 1] *= np.random.uniform(1-saturation, 1+saturation)  # Saturation
    hsv[:, :, 2] *= np.random.uniform(1-brightness, 1+brightness)  # Value
    
    # Clip values
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    # Convert back to RGB
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Adjust contrast
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * np.random.uniform(1-contrast, 1+contrast) + mean
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image