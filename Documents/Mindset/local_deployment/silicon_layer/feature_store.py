"""
Feature Store module for the Silicon Layer
Handles storage, versioning, and retrieval of features
"""
import os
import json
import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple

class FeatureStore:
    """
    Feature Store for managing ML features used in the Silicon Layer
    
    Capabilities:
    - Store features in Parquet format
    - Version features
    - Track feature lineage
    - Retrieve features by name, version, or date
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the feature store
        
        Args:
            base_dir: Base directory for feature storage
        """
        self.base_dir = Path(base_dir)
        self.features_dir = self.base_dir / "features"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create directories if they don't exist
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata file if it doesn't exist
        self.metadata_file = self.metadata_dir / "feature_metadata.json"
        if not self.metadata_file.exists():
            self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize empty metadata file"""
        metadata = {
            "features": {},
            "feature_sets": {},
            "last_updated": datetime.now().isoformat()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file"""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def _save_metadata(self, metadata: Dict):
        """Save metadata to file"""
        metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for versioning"""
        # Get a sample of the data to hash (for efficiency with large DataFrames)
        sample = df.head(1000) if len(df) > 1000 else df
        return hashlib.md5(pd.util.hash_pandas_object(sample).values).hexdigest()
    
    def store_features(
        self, 
        df: pd.DataFrame, 
        feature_name: str,
        description: str = None,
        tags: List[str] = None
    ) -> str:
        """
        Store features in the feature store
        
        Args:
            df: DataFrame containing features
            feature_name: Name of the feature set
            description: Description of the features
            tags: List of tags for the features
            
        Returns:
            version: Version string of the stored features
        """
        # Compute hash for versioning
        feature_hash = self._compute_hash(df)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        version = f"{timestamp}_{feature_hash[:8]}"
        
        # Create feature directory
        feature_dir = self.features_dir / feature_name
        feature_dir.mkdir(exist_ok=True)
        
        # Save features as Parquet
        file_path = feature_dir / f"{version}.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)
        
        # Update metadata
        metadata = self._load_metadata()
        
        if feature_name not in metadata["features"]:
            metadata["features"][feature_name] = {
                "versions": [],
                "description": description or "",
                "tags": tags or [],
                "created_at": datetime.now().isoformat()
            }
        
        version_info = {
            "version": version,
            "path": str(file_path.relative_to(self.base_dir)),
            "created_at": datetime.now().isoformat(),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": df.columns.tolist()
        }
        
        metadata["features"][feature_name]["versions"].append(version_info)
        metadata["features"][feature_name]["latest_version"] = version
        metadata["features"][feature_name]["last_updated"] = datetime.now().isoformat()
        
        self._save_metadata(metadata)
        
        return version
    
    def get_features(
        self, 
        feature_name: str, 
        version: str = "latest"
    ) -> pd.DataFrame:
        """
        Retrieve features from the feature store
        
        Args:
            feature_name: Name of the feature set
            version: Version to retrieve (default: "latest")
            
        Returns:
            DataFrame containing the features
        """
        metadata = self._load_metadata()
        
        if feature_name not in metadata["features"]:
            raise ValueError(f"Feature '{feature_name}' not found in feature store")
        
        feature_info = metadata["features"][feature_name]
        
        if version == "latest":
            version = feature_info["latest_version"]
        
        # Find version info
        version_info = None
        for v in feature_info["versions"]:
            if v["version"] == version:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version '{version}' not found for feature '{feature_name}'")
        
        # Load features from Parquet file
        file_path = self.base_dir / version_info["path"]
        return pd.read_parquet(file_path)
    
    def list_features(self) -> List[Dict]:
        """
        List all features in the feature store
        
        Returns:
            List of feature metadata
        """
        metadata = self._load_metadata()
        features = []
        
        for name, info in metadata["features"].items():
            features.append({
                "name": name,
                "description": info.get("description", ""),
                "tags": info.get("tags", []),
                "latest_version": info.get("latest_version"),
                "versions_count": len(info["versions"]),
                "last_updated": info.get("last_updated")
            })
        
        return features
    
    def create_feature_set(
        self,
        name: str,
        feature_names: List[str],
        description: str = None
    ) -> Dict:
        """
        Create a feature set from existing features
        
        Args:
            name: Name of the feature set
            feature_names: List of feature names to include
            description: Description of the feature set
            
        Returns:
            Feature set metadata
        """
        metadata = self._load_metadata()
        
        # Check if features exist
        for feature_name in feature_names:
            if feature_name not in metadata["features"]:
                raise ValueError(f"Feature '{feature_name}' not found in feature store")
        
        feature_set = {
            "name": name,
            "features": feature_names,
            "description": description or "",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        metadata["feature_sets"][name] = feature_set
        self._save_metadata(metadata)
        
        return feature_set
    
    def get_feature_set(self, name: str) -> Dict[str, pd.DataFrame]:
        """
        Retrieve a feature set
        
        Args:
            name: Name of the feature set
            
        Returns:
            Dictionary of feature DataFrames
        """
        metadata = self._load_metadata()
        
        if name not in metadata["feature_sets"]:
            raise ValueError(f"Feature set '{name}' not found")
        
        feature_set = metadata["feature_sets"][name]
        features = {}
        
        for feature_name in feature_set["features"]:
            features[feature_name] = self.get_features(feature_name)
        
        return features