#!/bin/bash
# Script to fix the EC2 instance setup

# Create directory structure
mkdir -p src/ml/silicon_layer
mkdir -p src/rust/metrics_engine
mkdir -p src/frontend

# Create basic files for the Silicon Layer
cat > src/ml/silicon_layer/__init__.py << 'EOF'
# Silicon Layer Module
EOF

cat > src/ml/silicon_layer.py << 'EOF'
"""
Silicon Layer: Advanced ML processing layer between Silver and Gold
"""

class SiliconLayer:
    def __init__(self, use_rust_engine=True):
        self.use_rust_engine = use_rust_engine
        print("Initializing Silicon Layer...")
        
    def process(self, data):
        """Process data through the Silicon Layer"""
        print("Processing data through Silicon Layer...")
        # Calculate metrics
        return self._calculate_metrics(data)
    
    def _calculate_metrics(self, data):
        """Calculate transparency metrics for news articles"""
        # This would normally use the Rust engine if available
        print(f"Calculating metrics with {'Rust' if self.use_rust_engine else 'Python fallback'}")
        
        # Mock metrics for demonstration
        for item in data:
            if 'metrics' not in item:
                item['metrics'] = {}
            
            # Political influence (0-10)
            item['metrics']['political_influence'] = 5.0
            
            # Rhetoric intensity (0-10)
            item['metrics']['rhetoric_intensity'] = 4.5
            
            # Information depth (0-10)
            item['metrics']['information_depth'] = 7.2
            
        return data
EOF

# Create Rust metrics engine placeholder
mkdir -p src/rust/metrics_engine/src

cat > src/rust/metrics_engine/Cargo.toml << 'EOF'
[package]
name = "metrics_engine"
version = "0.1.0"
edition = "2021"

[lib]
name = "metrics_engine_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.17.1", features = ["extension-module"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
regex = "1.7"
EOF

cat > src/rust/metrics_engine/src/lib.rs << 'EOF'
use pyo3::prelude::*;

/// Calculate article metrics
#[pyfunction]
fn calculate_metrics(article_text: &str) -> PyResult<(f64, f64, f64)> {
    // This is a simplified placeholder implementation
    // In a real system, this would analyze the text in depth
    
    // Political influence level (0-10)
    let political_influence = 5.0;
    
    // Rhetoric intensity (0-10)
    let rhetoric_intensity = 4.5;
    
    // Information depth (0-10)
    let information_depth = 7.2;
    
    Ok((political_influence, rhetoric_intensity, information_depth))
}

/// Python module
#[pymodule]
fn metrics_engine_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_metrics, m)?)?;
    Ok(())
}
EOF

cat > src/rust/metrics_engine/build.py << 'EOF'
#!/usr/bin/env python
"""
Build script for the Rust metrics engine
"""
import subprocess
import os
import sys

def main():
    print("Building Rust metrics engine...")
    
    # Check if we're just testing without Rust
    if "--skip-rust" in sys.argv:
        print("Skipping Rust build (test mode)")
        return 0
    
    # Try to build with maturin (preferred)
    try:
        result = subprocess.run(["maturin", "develop"], check=True)
        return result.returncode
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Maturin not available, trying cargo...")
    
    # Fall back to direct cargo build
    try:
        subprocess.run(["cargo", "build", "--release"], check=True)
        print("Rust build completed successfully")
        return 0
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Error building Rust component: {e}")
        print("Will use Python fallback methods")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

cat > src/rust/metrics_engine/py_wrapper.py << 'EOF'
"""
Python wrapper for the Rust metrics engine
"""

class MetricsEngine:
    """Wrapper for the Rust metrics calculation engine"""
    
    def __init__(self, use_rust=True):
        self.use_rust = use_rust
        self._rust_available = False
        
        # Try to import the Rust module
        if use_rust:
            try:
                import metrics_engine_rust
                self._rust_module = metrics_engine_rust
                self._rust_available = True
                print("Using Rust metrics engine")
            except ImportError:
                print("Rust metrics engine not available, using Python fallback")
    
    def calculate_metrics(self, article_text):
        """
        Calculate metrics for an article
        
        Returns:
            tuple: (political_influence, rhetoric_intensity, information_depth)
        """
        if self._rust_available:
            # Use the Rust implementation
            return self._rust_module.calculate_metrics(article_text)
        else:
            # Python fallback implementation
            return self._python_calculate_metrics(article_text)
    
    def _python_calculate_metrics(self, article_text):
        """
        Python fallback implementation for metrics calculation
        """
        # This is a simplified placeholder implementation
        # In a real system, this would analyze the text in depth
        
        # Political influence level (0-10)
        political_influence = 5.0
        
        # Rhetoric intensity (0-10)
        rhetoric_intensity = 4.5
        
        # Information depth (0-10)
        information_depth = 7.2
        
        return (political_influence, rhetoric_intensity, information_depth)
EOF

# Create a basic Frontend
mkdir -p src/frontend/pages

cat > src/frontend/package.json << 'EOF'
{
  "name": "mindset-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^12.3.1",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.1.3"
  }
}
EOF

cat > src/frontend/pages/index.js << 'EOF'
import React, { useState, useEffect } from 'react';

export default function Home() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Fetch articles from API
    fetch('/api/articles')
      .then(response => response.json())
      .then(data => {
        setArticles(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching articles:', error);
        setLoading(false);
      });
  }, []);
  
  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>MINDSET News Analytics</h1>
      <p>Transparent and explainable news analysis platform</p>
      
      {loading ? (
        <p>Loading articles...</p>
      ) : articles.length === 0 ? (
        <div>
          <p>No articles available yet. The system is still processing data.</p>
          <p>Check back in a few minutes after the pipeline has run.</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
          {articles.map(article => (
            <div key={article.id} style={{ 
              border: '1px solid #eaeaea', 
              borderRadius: '8px',
              padding: '15px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}>
              <h2>{article.title}</h2>
              <p>{article.description}</p>
              
              {article.metrics && (
                <div style={{ marginTop: '15px' }}>
                  <h3>Transparency Metrics</h3>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <div>
                      <strong>Political Influence:</strong> 
                      <div style={{ 
                        width: '100px', 
                        height: '10px', 
                        backgroundColor: '#eee',
                        marginTop: '5px'
                      }}>
                        <div style={{ 
                          width: `${article.metrics.political_influence * 10}%`, 
                          height: '100%', 
                          backgroundColor: '#ff6b6b'
                        }} />
                      </div>
                    </div>
                    
                    <div>
                      <strong>Rhetoric:</strong>
                      <div style={{ 
                        width: '100px', 
                        height: '10px', 
                        backgroundColor: '#eee',
                        marginTop: '5px'
                      }}>
                        <div style={{ 
                          width: `${article.metrics.rhetoric_intensity * 10}%`, 
                          height: '100%', 
                          backgroundColor: '#feca57'
                        }} />
                      </div>
                    </div>
                    
                    <div>
                      <strong>Depth:</strong>
                      <div style={{ 
                        width: '100px', 
                        height: '10px', 
                        backgroundColor: '#eee',
                        marginTop: '5px'
                      }}>
                        <div style={{ 
                          width: `${article.metrics.information_depth * 10}%`, 
                          height: '100%', 
                          backgroundColor: '#1dd1a1'
                        }} />
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              <a href={`/article/${article.id}`} style={{ 
                display: 'inline-block', 
                marginTop: '15px',
                color: '#2e86de',
                textDecoration: 'none'
              }}>
                Read more →
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
EOF

# Create a basic API
mkdir -p src/api

cat > src/api/main.py << 'EOF'
"""
MINDSET API - Main entry point
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from datetime import datetime
import random

app = FastAPI(title="MINDSET API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample articles for testing
SAMPLE_ARTICLES = [
    {
        "id": "article1",
        "title": "Understanding the Silicon Layer in News Analytics",
        "description": "How advanced ML processing improves transparency in news consumption",
        "content": "The Silicon Layer sits between the Silver and Gold data layers...",
        "source": "Tech News Daily",
        "url": "https://example.com/article1",
        "published_at": "2025-03-01T12:00:00Z",
        "metrics": {
            "political_influence": 3.2,
            "rhetoric_intensity": 2.8,
            "information_depth": 8.5
        }
    },
    {
        "id": "article2",
        "title": "Global Economic Outlook for 2025",
        "description": "Analysts predict steady growth despite ongoing challenges",
        "content": "Economic forecasters are predicting a measured but steady growth...",
        "source": "Financial Times",
        "url": "https://example.com/article2",
        "published_at": "2025-03-10T09:15:00Z",
        "metrics": {
            "political_influence": 5.1,
            "rhetoric_intensity": 3.0,
            "information_depth": 7.8
        }
    },
    {
        "id": "article3",
        "title": "New Climate Policy Announced",
        "description": "Government unveils ambitious targets for carbon reduction",
        "content": "The administration today announced sweeping new climate policies...",
        "source": "World News Network",
        "url": "https://example.com/article3",
        "published_at": "2025-03-12T16:30:00Z",
        "metrics": {
            "political_influence": 7.4,
            "rhetoric_intensity": 6.2,
            "information_depth": 5.9
        }
    }
]

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "MINDSET API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/api/articles")
async def get_articles():
    """Get all news articles with metrics"""
    try:
        # In a real implementation, this would retrieve articles from a database
        # For demonstration, return sample articles
        return SAMPLE_ARTICLES
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/articles/{article_id}")
async def get_article(article_id: str):
    """Get a specific article by ID"""
    try:
        # Find article by ID
        article = next((a for a in SAMPLE_ARTICLES if a["id"] == article_id), None)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return article
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics_summary():
    """Get summary of metrics across all articles"""
    try:
        # Calculate average metrics
        political_influence = sum(a["metrics"]["political_influence"] for a in SAMPLE_ARTICLES) / len(SAMPLE_ARTICLES)
        rhetoric_intensity = sum(a["metrics"]["rhetoric_intensity"] for a in SAMPLE_ARTICLES) / len(SAMPLE_ARTICLES)
        information_depth = sum(a["metrics"]["information_depth"] for a in SAMPLE_ARTICLES) / len(SAMPLE_ARTICLES)
        
        return {
            "count": len(SAMPLE_ARTICLES),
            "averages": {
                "political_influence": political_influence,
                "rhetoric_intensity": rhetoric_intensity,
                "information_depth": information_depth
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }
EOF

# Create a basic pipeline
cat > run_pipeline_aws.py << 'EOF'
#!/usr/bin/env python
"""
MINDSET AWS Data Pipeline
Processes data through Raw → Bronze → Silver → Silicon → Gold layers
"""
import os
import sys
import json
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset-pipeline')

def process_raw_to_bronze():
    """Process data from raw to bronze layer"""
    logger.info("Processing Raw → Bronze")
    # In a real implementation, this would:
    # 1. Read data from news API sources
    # 2. Store raw data in the bronze layer
    time.sleep(1)  # Simulate processing
    return {"status": "success", "articles_processed": 25}

def process_bronze_to_silver():
    """Process data from bronze to silver layer"""
    logger.info("Processing Bronze → Silver")
    # In a real implementation, this would:
    # 1. Clean and validate the data
    # 2. Extract features
    # 3. Apply transformations
    time.sleep(1)  # Simulate processing
    return {"status": "success", "articles_processed": 23}

def process_silver_to_silicon():
    """Process data from silver to silicon layer"""
    logger.info("Processing Silver → Silicon")
    
    # Import the Silicon Layer
    try:
        from src.ml.silicon_layer import SiliconLayer
        use_silicon = True
    except ImportError:
        logger.warning("Silicon Layer not available, skipping advanced metrics")
        use_silicon = False
    
    if use_silicon:
        try:
            # Check if we should use the Rust engine
            use_rust = os.environ.get('USE_RUST_ENGINE', 'true').lower() == 'true'
            
            # Initialize Silicon Layer
            silicon = SiliconLayer(use_rust_engine=use_rust)
            
            # Mock data that would come from Silver layer
            silver_data = [
                {"id": "article1", "title": "Test Article 1", "content": "Content 1"},
                {"id": "article2", "title": "Test Article 2", "content": "Content 2"},
                {"id": "article3", "title": "Test Article 3", "content": "Content 3"},
            ]
            
            # Process through Silicon Layer
            processed_data = silicon.process(silver_data)
            
            # Write to Silicon layer (in a real implementation)
            silicon_output = {
                "status": "success", 
                "articles_processed": len(processed_data),
                "metrics_calculated": True
            }
        except Exception as e:
            logger.error(f"Error in Silicon Layer processing: {e}")
            silicon_output = {"status": "error", "error": str(e)}
    else:
        silicon_output = {"status": "skipped", "reason": "Silicon Layer not available"}
    
    time.sleep(1)  # Simulate processing
    return silicon_output

def process_silicon_to_gold():
    """Process data from silicon to gold layer"""
    logger.info("Processing Silicon → Gold")
    # In a real implementation, this would:
    # 1. Add recommendations
    # 2. Prepare final data model for consumption
    # 3. Store in gold layer
    time.sleep(1)  # Simulate processing
    return {"status": "success", "articles_processed": 20}

def main():
    """Main pipeline execution function"""
    pipeline_start = datetime.now()
    logger.info(f"Starting MINDSET pipeline at {pipeline_start.isoformat()}")
    
    results = {
        "pipeline_id": f"pipeline-{pipeline_start.strftime('%Y%m%d-%H%M%S')}",
        "start_time": pipeline_start.isoformat(),
        "steps": {}
    }
    
    try:
        # Execute pipeline steps
        results["steps"]["raw_to_bronze"] = process_raw_to_bronze()
        results["steps"]["bronze_to_silver"] = process_bronze_to_silver()
        results["steps"]["silver_to_silicon"] = process_silver_to_silicon()
        results["steps"]["silicon_to_gold"] = process_silicon_to_gold()
        
        # Calculate overall status
        if all(step["status"] == "success" for step in results["steps"].values()):
            results["status"] = "success"
        else:
            results["status"] = "partial_success"
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        results["status"] = "error"
        results["error"] = str(e)
    
    # Record end time and duration
    pipeline_end = datetime.now()
    results["end_time"] = pipeline_end.isoformat()
    results["duration_seconds"] = (pipeline_end - pipeline_start).total_seconds()
    
    # Log results
    logger.info(f"Pipeline completed with status: {results['status']}")
    logger.info(f"Pipeline duration: {results['duration_seconds']} seconds")
    
    # In a real implementation, store the results to a log file or database
    with open("pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results["status"] == "success"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

# Make scripts executable
chmod +x run_pipeline_aws.py
chmod +x src/rust/metrics_engine/build.py

echo "Fixed MINDSET project structure created!"