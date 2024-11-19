import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf
from functools import partial


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CAMFeatureExtractor(nn.Module):
    """ResNet50 modified for Class Activation Mapping"""
    def __init__(self):
        super(CAMFeatureExtractor, self).__init__()
        # Load pretrained ResNet50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Get convolutional layers up to final conv layer
        self.features_conv = nn.Sequential(*list(resnet.children())[:-2])
        
        # Global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Save activation maps
        self.activation = None
        self.gradient = None
        
    def forward(self, x):
        x = self.features_conv(x)
        self.activation = x.detach()
        x = self.gap(x)
        return x.view(x.size(0), -1)

@dataclass
class OutlierDetectionConfig:
    """Configuration for outlier detection parameters"""
    contamination: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: Tuple[int, int] = (224, 224)
    methods: List[str] = None
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["isolation_forest"]

class ImageFolderDataset(Dataset):
    """Custom dataset for loading images from folder"""
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path

class FeatureExtractor:
    """Extract features from images using a pre-trained model with CAM support"""
    def get_lime_explanation(self, image_tensor: torch.Tensor, image_path: str) -> np.ndarray:
        """Generate LIME explanation for a single image"""
        # Convert tensor to numpy array and prepare for LIME
        image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
        
        def batch_predict(images):
            # Convert to torch tensor and move to device
            batch = torch.tensor(images.transpose(0, 3, 1, 2)).float().to(self.device)
            with torch.no_grad():
                return self.model(batch).cpu().numpy()
        
        # Initialize LIME image explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Get explanation
        explanation = explainer.explain_instance(
            image_np,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=1000,
            batch_size=self.config.batch_size
        )
        
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=True
        )
        
        # Create visualization
        lime_exp = mark_boundaries(temp, mask)
        return lime_exp
    def __init__(self, config: OutlierDetectionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load CAM-enabled model
        self.model = CAMFeatureExtractor().to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Store activation maps for CAM visualization
        self.activation_maps = {}

    def generate_cam(self, image_tensor: torch.Tensor, image_path: str) -> np.ndarray:
        """Generate class activation map for a single image"""
        # Get model activations
        with torch.no_grad():
            features = self.model(image_tensor.unsqueeze(0))
            activation = self.model.activation[0]  # Get activation for first (only) image
            
        # Calculate importance weights using GAP and ensure they're on the same device
        weights = torch.mean(activation, dim=(1, 2)).to(self.device)
        
        # Initialize CAM on the same device
        cam = torch.zeros(activation.shape[1:], device=self.device)
        
        # Generate CAM
        for i, w in enumerate(weights):
            cam += w * activation[i, :, :]
            
        # Move to CPU for numpy operations
        cam = cam.cpu()
            
        # Normalize CAM
        cam = cam.numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)  # Added small epsilon to avoid division by zero
        
        # Resize CAM to image size
        cam = cv2.resize(cam, self.config.image_size)
        
        self.activation_maps[image_path] = cam
        return cam

    def extract_features_batch(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from a batch of images"""
        dataset = ImageFolderDataset(image_paths, self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        features = []
        with torch.no_grad():
            for batch, paths in dataloader:
                batch = batch.to(self.device)
                batch_features = self.model(batch)
                features.append(batch_features.cpu().numpy())
                
                # Generate both CAM and LIME explanations for each image
                for img, path in zip(batch, paths):
                    self.generate_cam(img, path)
                    lime_exp = self.get_lime_explanation(img, path)
                    self.activation_maps[f"lime_{path}"] = lime_exp
        
        return np.vstack(features)
class OutlierDetector:
    """Detect outliers using multiple methods"""
    def __init__(self, config: OutlierDetectionConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        
    def detect_outliers(self, image_paths: List[str]) -> Dict[str, List[str]]:
        """Detect outliers using multiple methods and combine results"""
        features = self.feature_extractor.extract_features_batch(image_paths)
        features = StandardScaler().fit_transform(features)
        
        outliers = {method: [] for method in self.config.methods}
        
        for method in self.config.methods:
            if method == "isolation_forest":
                detector = IsolationForest(contamination=self.config.contamination, random_state=42)
            elif method == "elliptic_envelope":
                detector = EllipticEnvelope(contamination=self.config.contamination, random_state=42)
            
            predictions = detector.fit_predict(features)
            outlier_indices = np.where(predictions == -1)[0]
            outliers[method] = [image_paths[i] for i in outlier_indices]
            
        return outliers

    def visualize_outliers(self, outliers: Dict[str, List[str]], output_dir: Optional[str] = None):
        """Visualize detected outliers with CAM heatmaps and LIME explanations"""
        for method, outlier_paths in outliers.items():
            if not outlier_paths:
                logging.info(f"No outliers detected using {method}")
                continue
                
            n_outliers = len(outlier_paths)
            n_cols = min(5, n_outliers)
            n_rows = ((n_outliers - 1) // n_cols + 1) * 3  # Changed from 2 to 3 rows per image
            
            if n_outliers == 1:
                fig, axes = plt.subplots(3, 1, figsize=(5, 15))  # Changed from 2 to 3 rows
                axes = axes.reshape(3, 1)
            else:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9*((n_outliers-1)//n_cols + 1)))
                axes = axes.reshape(n_rows, n_cols)
            
            fig.suptitle(f'Outliers detected using {method}')
            
            for idx, img_path in enumerate(outlier_paths):
                col = idx % n_cols
                row = (idx // n_cols) * 3  # Changed from 2 to 3
                
                try:
                    # Original image
                    img = Image.open(img_path)
                    axes[row, col].imshow(img)
                    axes[row, col].axis('off')
                    axes[row, col].set_title(os.path.basename(img_path))
                    
                    # CAM heatmap
                    cam = self.feature_extractor.activation_maps[img_path]
                    img_array = np.array(img)
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = cv2.resize(heatmap, img_array.shape[1::-1])
                    
                    overlay = cv2.addWeighted(img_array, 0.7, heatmap, 0.3, 0)
                    axes[row+1, col].imshow(overlay)
                    axes[row+1, col].axis('off')
                    axes[row+1, col].set_title('CAM Heatmap')
                    
                    # LIME explanation
                    lime_exp = self.feature_extractor.activation_maps[f"lime_{img_path}"]
                    axes[row+2, col].imshow(lime_exp)
                    axes[row+2, col].axis('off')
                    axes[row+2, col].set_title('LIME Explanation')
                    
                except Exception as e:
                    logging.error(f"Error displaying image {img_path}: {str(e)}")
            
            # Turn off any unused subplots
            for row in range(0, n_rows, 3):
                for col in range(len(outlier_paths) % n_cols, n_cols):
                    if row // 3 * n_cols + col >= len(outlier_paths):
                        axes[row, col].axis('off')
                        axes[row+1, col].axis('off')
                        axes[row+2, col].axis('off')
            
            plt.tight_layout()
            
            if output_dir:
                output_path = os.path.join(output_dir, f'outliers_cam_lime_{method}.png')
                try:
                    plt.savefig(output_path)
                    logging.info(f"Saved outlier visualization to {output_path}")
                except Exception as e:
                    logging.error(f"Error saving visualization to {output_path}: {str(e)}")
            else:
                plt.show()
            
            plt.close()

def process_dataset(data_path: str, config: OutlierDetectionConfig, output_dir: Optional[str] = None):
    """Process entire dataset and detect outliers in each class"""
    data_path = Path(data_path)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    detector = OutlierDetector(config)
    results = {}

    for class_folder in data_path.iterdir():
        if not class_folder.is_dir():
            continue
            
        logging.info(f"Processing class: {class_folder.name}")
        
        # Get all image paths for the current class
        image_paths = [
            str(p) for p in class_folder.iterdir()
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ]
        
        if not image_paths:
            logging.warning(f"No images found in {class_folder}")
            continue
            
        # Detect outliers
        outliers = detector.detect_outliers(image_paths)
        results[class_folder.name] = outliers
        
        # Visualize results
        if output_dir:
            class_output_dir = output_dir / class_folder.name
            class_output_dir.mkdir(exist_ok=True)
            detector.visualize_outliers(outliers, str(class_output_dir))
        else:
            detector.visualize_outliers(outliers)
            
    return results

if __name__ == "__main__":
    # Example usage
    config = OutlierDetectionConfig(
        contamination=0.1,
        batch_size=32,
        methods=["isolation_forest"],
        image_size=(224, 224)
    )
    
    # Update these paths to your actual data and output directories
    data_path = r"C:\Users\gravy\Downloads\FaceRecognitionDataCleaningWithXAI-main\FaceRecognitionDataCleaningWithXAI-main\Sample1"
    output_dir = r"C:\Users\gravy\Downloads\FaceRecognitionDataCleaningWithXAI-main\FaceRecognitionDataCleaningWithXAI-main\Sample_new"
    
    results = process_dataset(data_path, config, output_dir)