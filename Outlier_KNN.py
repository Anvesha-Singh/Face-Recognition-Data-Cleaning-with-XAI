import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class OutlierDetectionConfig:
    """Configuration for outlier detection parameters"""
    contamination: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: Tuple[int, int] = (224, 224)
    methods: List[str] = None
    n_neighbors: int = 5  # Number of neighbors for KNN
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["knn"]

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
    """Extract features from images using a pre-trained model"""
    def __init__(self, config: OutlierDetectionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load ResNet50 model
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove classification layer
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
            for batch, _ in dataloader:
                batch = batch.to(self.device)
                batch_features = self.model(batch).squeeze()
                features.append(batch_features.cpu().numpy())
                
        return np.vstack(features) if len(features[0].shape) == 2 else np.concatenate(features)

class OutlierDetector:
    """Detect outliers using multiple methods, including KNN"""
    def __init__(self, config: OutlierDetectionConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        
    def detect_outliers(self, image_paths: List[str]) -> Dict[str, List[str]]:
        """Detect outliers using multiple methods and combine results"""
        features = self.feature_extractor.extract_features_batch(image_paths)
        features = StandardScaler().fit_transform(features)
        
        outliers = {method: [] for method in self.config.methods}
        
        for method in self.config.methods:
            if method == "knn":
                knn = NearestNeighbors(n_neighbors=self.config.n_neighbors)
                knn.fit(features)
                
                distances, _ = knn.kneighbors(features)
                threshold = np.percentile(distances[:, -1], (1 - self.config.contamination) * 100)
                outlier_indices = np.where(distances[:, -1] > threshold)[0]
                
                outliers[method] = [image_paths[i] for i in outlier_indices]
                
        return outliers

    def visualize_outliers(self, outliers: Dict[str, List[str]], output_dir: Optional[str] = None):
        """Visualize detected outliers"""
        for method, outlier_paths in outliers.items():
            if not outlier_paths:
                logging.info(f"No outliers detected using {method}")
                continue
                
            n_outliers = len(outlier_paths)
            n_cols = min(5, n_outliers)
            n_rows = (n_outliers - 1) // n_cols + 1
            
            if n_outliers == 1:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                axes = [ax]
            else:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
                axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            
            fig.suptitle(f'Outliers detected using {method}')
            
            for idx, img_path in enumerate(outlier_paths):
                try:
                    img = Image.open(img_path)
                    axes[idx].imshow(img)
                    axes[idx].axis('off')
                    axes[idx].set_title(os.path.basename(img_path))
                except Exception as e:
                    logging.error(f"Error displaying image {img_path}: {str(e)}")
            
            for idx in range(len(outlier_paths), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            if output_dir:
                output_path = os.path.join(output_dir, f'outliers_{method}.png')
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
        
        image_paths = [
            str(p) for p in class_folder.iterdir()
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ]
        
        if not image_paths:
            logging.warning(f"No images found in {class_folder}")
            continue
            
        outliers = detector.detect_outliers(image_paths)
        results[class_folder.name] = outliers
        
        if output_dir:
            class_output_dir = output_dir / class_folder.name
            class_output_dir.mkdir(exist_ok=True)
            detector.visualize_outliers(outliers, str(class_output_dir))
        else:
            detector.visualize_outliers(outliers)
            
    return results

if __name__ == "__main__":
    config = OutlierDetectionConfig(
        contamination=0.1,
        batch_size=32,
        methods=["knn"],
        image_size=(224, 224),
        n_neighbors=5
    )
    
    data_path = r"C:\Users\gravy\Downloads\FaceRecognitionDataCleaningWithXAI-main\FaceRecognitionDataCleaningWithXAI-main\Sample"
    output_dir = r"C:\Users\gravy\Downloads\FaceRecognitionDataCleaningWithXAI-main\FaceRecognitionDataCleaningWithXAI-main\Sample2"
    
    results = process_dataset(data_path, config, output_dir)

