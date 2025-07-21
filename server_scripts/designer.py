import os
import json
import uuid
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient, models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
import torch
from PIL import Image
import io
import tempfile
import shutil

# CLIP imports  Install with: pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("⚠CLIP not available. Install with: pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False

# Configuration
QDRANT_URL = 
QDRANT_API_KEY = 
QDRANT_COLLECTION_NAME = "fashion_clip_recommender"
VECTOR_SIZE = 512
NAMESPACE_UUID = uuid.NAMESPACE_DNS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOTHES_PATH = os.path.join(BASE_DIR, "clothes_tryon_dataset", "test", "cloth")
METADATA_DIR = os.path.join(BASE_DIR, "features_extracted")

# Pydantic models
class VisualSearchRequest(BaseModel):
    limit: int = 10

class VisualSearchResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    total_found: int
    query_info: Dict[str, Any]
    similarity_scores: List[float]
    error: str | None = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    print("🚀 Starting Fashion Visual Designer API...")
    
    clip_ready = setup_clip()

    qdrant_ready = setup_qdrant_collection()
    
    if clip_ready and qdrant_ready:
        print("All services initialized successfully!")
    else:
        print("⚠Some services failed to initialize")
    
    yield
    

    print("Shutting down Fashion Visual Designer API...")

# FastAPI app setup
app = FastAPI(
    title="Fashion Visual Designer API", 
    description="CLIPpowered visual similarity search for fashion items",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving images
app.mount("/static/clothes", StaticFiles(directory=CLOTHES_PATH), name="clothes")

# Initialize clients
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# CLIP model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = None
clip_preprocess = None

def setup_clip():
    """Initialize CLIP model"""
    global clip_model, clip_preprocess
    
    if not CLIP_AVAILABLE:
        print("CLIP not available  visual search will be disabled")
        return False
    
    try:
        print(f"Loading CLIP model on device: {device}")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print("CLIP model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return False

def get_clip_embedding(image: Image.Image) -> np.ndarray:
    """Extract CLIP embedding from PIL Image"""
    if not CLIP_AVAILABLE or clip_model is None:
        raise HTTPException(status_code=500, detail="CLIP model not available")
    
    try:
        # Preprocess image for CLIP
        image_tensor = clip_preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get image features
            image_features = clip_model.encode_image(image_tensor)
            # Normalize features
            image_features /= image_features.norm(dim=1, keepdim=True)
            
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting CLIP embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def setup_qdrant_collection():
    """Creates the CLIP collection if it doesn't exist"""
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if QDRANT_COLLECTION_NAME in collection_names:
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists")
            return True
    except Exception as e:
        print(f"Could not check existing collections: {e}")
    
    try:
        print(f"🔧 Creating collection '{QDRANT_COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        
        print("Collection setup completed")
        return True
    except UnexpectedResponse as e:
        if "already exists" in str(e).lower():
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists")
            return True
        else:
            print(f"Error creating collection: {e}")
            return False
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False

def find_similar_items(query_vector: np.ndarray, limit: int = 10) -> List[Dict[str, Any]]:
    """Find similar items using CLIP vector similarity"""
    try:
        print(f"Searching Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
        print(f"Query vector shape: {query_vector.shape}, Limit: {limit}")
        
        # Search for similar vectors
        search_results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=limit,
            with_payload=True
        )
        
        print(f"Qdrant returned {len(search_results)} results")
        
        results = []
        for i, result in enumerate(search_results):
            original_id = result.payload.get('original_id', 'N/A')
            score = float(result.score)
            print(f"  #{i+1}: ID={original_id}, Score={score:.4f}")
            
            results.append({
                "id": i + 1,
                "image_id": original_id,
                "name": f"Fashion Item {original_id}",
                "category": result.payload.get('clothing_type', 'Fashion Item'),
                "image": f"http://localhost:8004/static/clothes/{original_id}.jpg",
                "similarity_score": score,
                "dominant_color": result.payload.get('dominant_color', 'Unknown'),
                "sleeve_type": result.payload.get('sleeve_type', 'Unknown'),
                "pattern_type": result.payload.get('pattern_type', 'Unknown'),
                "description": f"{result.payload.get('clothing_type', 'Fashion item')} with {result.payload.get('pattern_type', 'classic')} pattern",
                "metadata": result.payload
            })
        
        print(f"Processed {len(results)} results successfully")
        return results
    except Exception as e:
        print(f"Error searching similar items: {e}")
        import traceback
        traceback.print_exc()
        return []

def populate_clip_database():
    """Populate database with CLIP embeddings from existing images"""
    if not CLIP_AVAILABLE or clip_model is None:
        print("CLIP not available  cannot populate database")
        return False
    
    print("🔄 Starting CLIP database population...")
    
    try:
        # Get list of existing images
        image_files = [f for f in os.listdir(CLOTHES_PATH) if f.endswith('.jpg')]
        
        points_to_upsert = []
        processed = 0
        
        for image_file in image_files:
            try:
                image_id = os.path.splitext(image_file)[0]
                image_path = os.path.join(CLOTHES_PATH, image_file)
                metadata_path = os.path.join(METADATA_DIR, f"{image_id}.json")
                
                # Load image and get CLIP embedding
                image = Image.open(image_path)
                vector = get_clip_embedding(image)
                
                # Load metadata if available
                metadata = {"original_id": image_id}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata.update(json.load(f))
                
                # Create point
                point_id = str(uuid.uuid5(NAMESPACE_UUID, image_id))
                points_to_upsert.append(
                    qdrant_models.PointStruct(
                        id=point_id, 
                        vector=vector.tolist(), 
                        payload=metadata
                    )
                )
                
                processed += 1
                
                # Batch upsert
                if len(points_to_upsert) >= 50:
                    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_to_upsert, wait=True)
                    points_to_upsert = []
                    print(f"📊 Processed {processed} images...")
                
                # Limit for demo
                if processed >= 100:  # Process first 100 images for demo
                    break
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        # Final batch
        if points_to_upsert:
            client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_to_upsert, wait=True)
        
        print(f"Database population complete! Processed {processed} images.")
        return True
        
    except Exception as e:
        print(f"Error populating database: {e}")
        return False

# FastAPI endpoints
@app.get("/")
async def root():
    return {
        "message": "Fashion Visual Designer API", 
        "features": ["CLIPpowered visual search", "Image similarity", "Fashion recommendations"],
        "endpoints": ["/visualsearch", "/health", "/populatesample", "/debugcollection", "/collectionstats", "/docs"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Qdrant connection
        collections = client.get_collections()
        collection_exists = QDRANT_COLLECTION_NAME in [col.name for col in collections.collections]
        
        # Check collection size
        collection_info = None
        if collection_exists:
            collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
        
        return {
            "status": "healthy",
            "clip_available": CLIP_AVAILABLE and clip_model is not None,
            "qdrant_connection": "ok",
            "collection_exists": collection_exists,
            "collection_size": collection_info.points_count if collection_info else 0,
            "device": str(device)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "clip_available": False,
            "qdrant_connection": "error"
        }

@app.post("/visualsearch")
async def visual_search(file: UploadFile = File(...), limit: int = 10):
    """Upload an image and find visually similar fashion items"""
    print(f"Visual search request received  File: {file.filename}, Limit: {limit}")
    
    if not CLIP_AVAILABLE or clip_model is None:
        print("CLIP model not available")
        raise HTTPException(status_code=500, detail="CLIP model not available")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        print(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process uploaded image
        print("📖 Reading uploaded image...")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        print(f"Image loaded  Size: {image.size}, Mode: {image.mode}")
        
        # Get CLIP embedding
        print("🧠 Extracting CLIP embedding...")
        query_vector = get_clip_embedding(image)
        print(f"CLIP embedding extracted  Vector shape: {query_vector.shape}")
        
        # Find similar items
        print("🔍 Searching for similar items...")
        similar_items = find_similar_items(query_vector, limit)
        print(f"Found {len(similar_items)} similar items")
        
        if not similar_items:
            print("No similar items found")
            return VisualSearchResponse(
                recommendations=[],
                total_found=0,
                query_info={"filename": file.filename, "size": list(image.size)},
                similarity_scores=[],
                error="No similar items found. Database might be empty."
            )
        
        print("Returning search results")
        return VisualSearchResponse(
            recommendations=similar_items,
            total_found=len(similar_items),
            query_info={
                "filename": file.filename,
                "size": list(image.size),
                "format": image.format
            },
            similarity_scores=[item["similarity_score"] for item in similar_items],
            error=None
        )
        
    except Exception as e:
        print(f"Error in visual search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Alias endpoint for hyphenated URL (for frontend compatibility)
@app.post("/visual-search")
async def visual_search_alias(file: UploadFile = File(...), limit: int = 10):
    """Alias for /visualsearch endpoint - Upload an image and find visually similar fashion items"""
    return await visual_search(file, limit)

@app.get("/randomitems")
async def get_random_items(limit: int = 10):
    """Get random fashion items from the database"""
    try:
        import random
        
        # Get random sample from collection
        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=limit * 2,  # Get more to ensure variety
            with_payload=True
        )
        
        if not points:
            return {
                "items": [],
                "total_count": 0,
                "error": "No items found in database"
            }
        
        # Randomly select items
        random_points = random.sample(points, min(limit, len(points)))
        
        random_items = []
        for point in random_points:
            item = {
                "id": point.payload.get("original_id", "unknown"),
                "name": f"Fashion Item {point.payload.get('original_id', 'Unknown')}",
                "category": point.payload.get("clothing_type", "Unknown"),
                "image": f"http://localhost:8004/static/clothes/{point.payload.get('original_id', 'unknown')}.jpg",
                "dominant_color": point.payload.get("dominant_color", "Unknown"),
                "sleeve_type": point.payload.get("sleeve_type", "Unknown"),
                "pattern_type": point.payload.get("pattern_type", "Unknown"),
                "description": f"{point.payload.get('clothing_type', 'Fashion item')} with {point.payload.get('pattern_type', 'classic')} pattern"
            }
            random_items.append(item)
        
        return {
            "items": random_items,
            "total_count": len(random_items),
            "error": None
        }
        
    except Exception as e:
        print(f"Error fetching random items: {e}")
        return {
            "items": [],
            "total_count": 0,
            "error": str(e)
        }

@app.post("/populatesample")
async def populate_sample():
    """Populate the database with a small sample of CLIP embeddings for testing"""
    if not CLIP_AVAILABLE or clip_model is None:
        raise HTTPException(status_code=500, detail="CLIP model not available")
    
    try:
        print("Starting sample CLIP database population...")
        
        image_files = [f for f in os.listdir(CLOTHES_PATH) if f.endswith('.jpg')][:5]
        
        if not image_files:
            return {"error": "No image files found in the clothes directory"}
        
        points_to_upsert = []
        processed = 0
        
        for image_file in image_files:
            try:
                image_id = os.path.splitext(image_file)[0]
                image_path = os.path.join(CLOTHES_PATH, image_file)
                metadata_path = os.path.join(METADATA_DIR, f"{image_id}.json")
                
                print(f"Processing {image_file}...")
                
                # Load image and get CLIP embedding
                image = Image.open(image_path)
                vector = get_clip_embedding(image)
                
                # Load metadata if available
                metadata = {"original_id": image_id}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata.update(json.load(f))
                
                # Create point
                point_id = str(uuid.uuid5(NAMESPACE_UUID, image_id))
                points_to_upsert.append(
                    qdrant_models.PointStruct(
                        id=point_id, 
                        vector=vector.tolist(), 
                        payload=metadata
                    )
                )
                
                processed += 1
                print(f"Processed {image_file}")
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        # Upsert all points
        if points_to_upsert:
            client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_to_upsert, wait=True)
            print(f"Sample database population complete! Processed {processed} images.")
            
            return {
                "status": "success",
                "message": f"Sample database populated with {processed} images",
                "processed_images": [os.path.splitext(f)[0] for f in image_files[:processed]]
            }
        else:
            return {"error": "No images could be processed"}
        
    except Exception as e:
        print(f"Error populating sample database: {e}")
        raise HTTPException(status_code=500, detail=f"Error populating database: {str(e)}")

@app.get("/collectionstats")
async def get_collection_stats():
    """Get statistics about the CLIP collection"""
    try:
        collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
        
        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=100,
            with_payload=True
        )
        
        categories = {}
        for point in points:
            category = point.payload.get('clothing_type', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_items": collection_info.points_count,
            "vector_size": VECTOR_SIZE,
            "categories": categories,
            "top_categories": sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection stats: {str(e)}")

@app.get("/debugcollection")
async def debug_collection():
    """Debug endpoint to check collection status"""
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        collection_exists = QDRANT_COLLECTION_NAME in collection_names
        
        result = {
            "collection_name": QDRANT_COLLECTION_NAME,
            "collection_exists": collection_exists,
            "all_collections": collection_names,
            "vector_size": VECTOR_SIZE
        }
        
        if collection_exists:
            try:
                # Get a few sample points
                points, _ = client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    limit=3,
                    with_payload=True
                )
                
                result["sample_count"] = len(points)
                result["sample_points"] = []
                for point in points:
                    result["sample_points"].append({
                        "id": str(point.id),
                        "original_id": point.payload.get("original_id", "N/A"),
                        "payload_keys": list(point.payload.keys()) if point.payload else []
                    })
            except Exception as e:
                result["scroll_error"] = str(e)
        
        return result
    except Exception as e:
        return {"error": str(e), "collection_name": QDRANT_COLLECTION_NAME}

def main():
    """Standalone mode for testing"""
    print("=== Fashion Visual Designer ===")
    print("CLIPpowered visual similarity search")
    print()
    
    # Setup services
    clip_ready = setup_clip()
    qdrant_ready = setup_qdrant_collection()
    
    if not clip_ready:
        print("❌ CLIP not available. Please install: pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git")
        return
    
    if not qdrant_ready:
        print("❌ Qdrant connection failed")
        return
    
    print("All services ready!")
    print("Use the API endpoints for visual search functionality")

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Fashion Visual Designer API on port 8004...")
    print("CLIP-powered visual similarity search")
    print("Static files: http://localhost:8004/static/clothes/")
    print("API docs: http://localhost:8004/docs")
    print("Features: Upload image → Find similar fashion items")
    uvicorn.run(app, host="0.0.0.0", port=8004)
