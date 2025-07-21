import os
import json
import uuid
import numpy as np
import tempfile
import shutil
import io
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient, models as qdrant_models
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
from pydantic import BaseModel

# Import AI Search functionality
import google.generativeai as genai
from ast import literal_eval
from itertools import combinations
import hashlib
import time

QDRANT_URL = 
QDRANT_API_KEY = 
QDRANT_COLLECTION_NAME = "fashion_visual_recommender"
VECTOR_SIZE = 4
NAMESPACE_UUID = uuid.NAMESPACE_DNS

# Get the parent directory for data files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOTHES_PATH = os.path.join(BASE_DIR, "clothes_tryon_dataset", "test", "cloth")
CLASS_MAPPING_DIR = BASE_DIR

# Configure Gemini API
genai.configure(api_key="")

llm_cache = {}
cache_max_size = 100  # Limit cache size to prevent memory issues

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    offset: int = 0  # For pagination in continuous search

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    print("Fashion Recommender API started successfully!")
    print("Qdrant collection 'fashion_metadata_store' is ready for use.")
    
    yield
    
    # Shutdown (if needed)
    print("👋 Shutting down Fashion Recommender API...")

app = FastAPI(title="Fashion Recommender API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/clothes", StaticFiles(directory=CLOTHES_PATH), name="clothes")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Load class mapping files from base directory
    class_file = os.path.join(CLASS_MAPPING_DIR, "class2idx.json")
    subclass_file = os.path.join(CLASS_MAPPING_DIR, "subclass2idx.json")
    
    with open(class_file) as f:
        class2idx = json.load(f)
    with open(subclass_file) as f:
        subclass2idx = json.load(f)
        
except FileNotFoundError as e:
    print(f"Required files not found: {e}")
    print("Please ensure class2idx.json and subclass2idx.json are in the base directory")
    exit()

idx2class = {int(v): k for k, v in class2idx.items()}
idx2subclass = {int(v): k for k, v in subclass2idx.items()}

class MultiOutputResNet(nn.Module):
    def __init__(self, num_classes, num_subclasses):
        super().__init__()
        self.base = models.resnet18(weights=None)
        self.base.fc = nn.Identity()
        self.fc_class = nn.Linear(512, num_classes)
        self.fc_subclass = nn.Linear(512, num_subclasses)

    def forward(self, x):
        features = self.base(x)
        out_class = self.fc_class(features)
        out_subclass = self.fc_subclass(features)
        return out_class, out_subclass

num_classes = len(class2idx)
num_subclasses = len(subclass2idx)
color_model = MultiOutputResNet(num_classes, num_subclasses)

try:
    model_path = os.path.join(BASE_DIR, "model_weights.pth")
    color_model.load_state_dict(torch.load(model_path, map_location=device))
    color_model.to(device)
    color_model.eval()
    print("Model loaded successfully")
except FileNotFoundError:
    print("Warning: Could not load model weights")
    print("Face color analysis features will be limited")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_translation = {
    "autunno": "autumn",
    "estate": "summer",
    "inverno": "winter",
    "primavera": "spring"
}

palette_colors = {
    "deep autumn": ["Brown", "Burgundy"],
    "soft autumn": ["Beige", "Peach"],
    "warm autumn": ["Orange", "Gold"],
    "cool summer": ["Blue", "Mint"],
    "light summer": ["Aqua", "Pink"],
    "soft summer": ["Gray", "Lavender"],
    "bright winter": ["Red", "Black"],
    "cool winter": ["Navy", "Purple"],
    "deep winter": ["Black", "Teal"],
    "bright spring": ["Yellow", "Coral"],
    "light spring": ["Peach", "Aqua"],
    "warm spring": ["Gold", "Pink"],
}

def setup_qdrant_collection():
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if QDRANT_COLLECTION_NAME in collection_names:
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")
            return
    except Exception as e:
        print(f"Could not check existing collections: {e}")
    
    try:
        print(f"Creating collection '{QDRANT_COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(size=VECTOR_SIZE, distance=qdrant_models.Distance.COSINE),
        )
        
        filterable_fields = [
            "clothing_type", "gender_suitability", "occasion_suitability",
            "sleeve_type", "neckline", "closure_type", "dominant_color",
            "secondary_color", "pattern_type", "pattern_description"
        ]
        for field in filterable_fields:
            try:
                client.create_payload_index(
                    collection_name=QDRANT_COLLECTION_NAME,
                    field_name=field,
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                    wait=True
                )
                print(f"Created index for '{field}'")
            except Exception as e:
                print(f"Index for '{field}' might already exist: {e}")
        print("Collection setup completed.")
    except Exception as e:
        print(f"Error creating collection: {e}")
        print("Collection might already exist or there might be a connection issue.")

def ensure_collection_exists():
    """Ensure the Qdrant collection exists before performing operations"""
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if QDRANT_COLLECTION_NAME not in collection_names:
            setup_qdrant_collection()
    except Exception as e:
        print(f"Error ensuring collection exists: {e}")

def search_by_metadata(filters: dict, limit: int = 10) -> list:
    """Optimized metadata search with better performance"""
    
    # PERFORMANCE OPTIMIZATION: Skip empty filters early
    if not filters:
        return []
    
    must_conditions = []
    for key, value in filters.items():
        if value is None or str(value).strip() == '' or str(value).lower() == 'n/a':
            continue

        if key in ["gender_suitability", "occasion_suitability"]:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchAny(any=[value])
                )
            )
        else:
            must_conditions.append(
                qdrant_models.FieldCondition(key=key, match=qdrant_models.MatchValue(value=value))
            )

    # PERFORMANCE OPTIMIZATION: Skip search if no valid conditions
    if not must_conditions:
        return []

    try:
        search_results, _ = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=qdrant_models.Filter(
                must=must_conditions
            ),
            limit=limit,
            with_payload=True,
        )
        return search_results
    except Exception as e:
        print(f"Error searching metadata: {e}")
        return []

def predict_colors_from_uploaded_image(image: Image.Image) -> tuple:
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            out_class, out_subclass = color_model(image_tensor)
            pred_class_idx = torch.argmax(out_class, dim=1).item()
            pred_subclass_idx = torch.argmax(out_subclass, dim=1).item()

        pred_class_name = idx2class.get(pred_class_idx, f"Unknown class index: {pred_class_idx}")
        pred_subclass_name = idx2subclass.get(pred_subclass_idx, f"Unknown subclass index: {pred_subclass_idx}")

        english_class = class_translation.get(pred_class_name.lower(), pred_class_name)
        combined_palette = f"{pred_subclass_name.lower()} {english_class.lower()}"

        dominant_color, secondary_color = get_palette_colors(combined_palette, palette_colors)

        return dominant_color, secondary_color, combined_palette

    except Exception as e:
        print(f"Error in color prediction: {e}")
        return None, None, None

def get_palette_colors(predicted_palette: str, palette_dict: dict) -> tuple:
    color_list = palette_dict.get(predicted_palette.lower(), [])
    dominant_color = color_list[0] if len(color_list) > 0 else None
    secondary_color = color_list[1] if len(color_list) > 1 else None
    return dominant_color, secondary_color

def get_clothing_details_by_ids(image_ids: List[str]) -> List[Dict[str, Any]]:
    clothing_details = []
    for i, image_id in enumerate(image_ids):
        image_path = os.path.join(CLOTHES_PATH, f"{image_id}.jpg")
        if os.path.exists(image_path):
            clothing_details.append({
                "id": i + 1,
                "image_id": image_id,
                "name": f"Recommended Item {image_id}",
                "category": "Fashion Item",
                "image": f"http://localhost:8001/static/clothes/{image_id}.jpg",
                "matchScore": max(85, 95 - (i * 2)),
                "reason": "AI-selected based on your color palette and style"
            })
    return clothing_details

def get_clothing_recommendations_by_uploaded_image(image: Image.Image, limit: int = 10) -> Dict[str, Any]:
    dominant_color, secondary_color, palette_type = predict_colors_from_uploaded_image(image)
    
    # If color prediction failed, return error
    if dominant_color is None:
        return {
            "error": "Unable to detect clear facial features for color analysis. Please provide a clear, well-lit image of your face without heavy filters or obstructions.",
            "recommendations": [],
            "theme": None
        }
    
    search_results = []
    
    if dominant_color:
        filters = {'dominant_color': dominant_color}
        search_results = search_by_metadata(filters=filters, limit=limit)
    
    if not search_results and secondary_color:
        filters = {'dominant_color': secondary_color}
        search_results = search_by_metadata(filters=filters, limit=limit)

    image_ids = []
    for result in search_results:
        original_id = result.payload.get('original_id', 'N/A')
        if original_id != 'N/A':
            image_ids.append(original_id)
    
    clothing_details = get_clothing_details_by_ids(image_ids[:limit])
    
    # Create theme message
    theme_colors = palette_colors.get(palette_type.lower(), [dominant_color, secondary_color])
    theme_message = f"Today you have the aura of {palette_type.title()}. {', '.join(theme_colors)} will suit you perfectly today!"
    
    return {
        "recommendations": clothing_details,
        "theme": theme_message,
        "error": None
    }

def extract_attributes_with_gemini(user_prompt: str) -> dict:
    """Extract clothing attributes using optimized Gemini LLM prompt with caching"""
    
    # PERFORMANCE OPTIMIZATION: Check cache first
    cache_key = hashlib.md5(user_prompt.lower().strip().encode()).hexdigest()
    
    if cache_key in llm_cache:
        print(f"Cache hit for query: {user_prompt}")
        return llm_cache[cache_key]
    
    # Start timing
    start_time = time.time()
    
    llm_instruction = f
    """
Extract fashion attributes from: "{user_prompt}"

CRITICAL: Extract ALL explicitly mentioned attributes. Pay special attention to:
- "full sleeve" = Long Sleeve
- "long sleeve" = Long Sleeve  
- "short sleeve" = Short Sleeve
- Color names should match exactly from the list

ATTRIBUTES (extract if mentioned):
- clothing_type: [Blouse, Shirt, T-shirt, Hoodie, Sweater, Blazer, Jacket, Dress, Skirt, Pants, Shorts, Jeans, Coat, Vest, Tank Top, Cardigan, Jumpsuit, Romper, Leggings, Tunic]
- sleeve_type: [Sleeveless, Long Sleeve, Short Sleeve, Three-Quarter Sleeve, Cap Sleeve, Bell Sleeve, Puff Sleeve]
- neckline: [Collar, Crew Neck, V-Neck, High Neck, Hooded, Scoop Neck, Strapless, Off-Shoulder, Boat Neck, Cowl Neck, Halter, Square Neck]
- closure_type: [None, Zipper, Tie, Button, Velcro, Pullover, Wrap, Drawstring, Snap, Hook, Lace-up]
- pattern_type: [Striped, Graphic Print, Solid, Floral, Polka Dot, Animal Print, Checked, Plaid, Geometric, Abstract, Embroidered, Paisley, Tie-Dye, Leopard, Zebra]
- gender_suitability: [Male, Female, Unisex, Child]
- occasion_suitability: [Casual, Formal, Party Wear, Sportswear, Business Casual, Beachwear, Lounge Wear, Work Wear, Evening Wear, Wedding]
- dominant_color: [Red, Blue, Green, Yellow, Black, White, Brown, Gray, Pink, Purple, Orange, Teal, Mint, Aqua, Gold, Peach, Burgundy, Lavender, Navy, Beige, Coral, Turquoise, Maroon, Olive, Cream]
- secondary_color: [Red, Blue, Green, Yellow, Black, White, Brown, Gray, Pink, Purple, Orange, Teal, Mint, Aqua, Gold, Peach, Burgundy, Lavender, Navy, Beige, Coral, Turquoise, Maroon, Olive, Cream]

EXAMPLES:
- "full sleeve floral dress blue colour" → {{"clothing_type": "Dress", "sleeve_type": "Long Sleeve", "pattern_type": "Floral", "dominant_color": "Blue"}}
- "red shirt" → {{"clothing_type": "Shirt", "dominant_color": "Red"}}

Create ALL possible search combinations (individual, pairs, triples, full).

RESPOND WITH JSON ONLY:
{{
    "extracted_attributes": {{"attr": "value", ...}},
    "search_combinations": [
        {{"combination_type": "individual", "filters": {{"attr1": "val1"}}}},
        {{"combination_type": "pair", "filters": {{"attr1": "val1", "attr2": "val2"}}}},
        {{"combination_type": "full", "filters": {{"attr1": "val1", "attr2": "val2", "attr3": "val3"}}}}
    ]
}}
"""

    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content(llm_instruction)

    text_out = response.text.strip()
    llm_time = time.time() - start_time
    print(f"LLM call took {llm_time:.2f}s")

    # Clean response
    if text_out.startswith('```'):
        text_out = '\n'.join([line for line in text_out.splitlines() if not line.strip().startswith('```')])

    try:
        result = json.loads(text_out)
        
        # PERFORMANCE OPTIMIZATION: Cache the result
        if len(llm_cache) >= cache_max_size:
            # Remove oldest entry if cache is full
            oldest_key = next(iter(llm_cache))
            del llm_cache[oldest_key]
        
        llm_cache[cache_key] = result
        return result
        
    except json.JSONDecodeError:
        try:
            result = literal_eval(text_out)
            
            # Cache successful result
            if len(llm_cache) >= cache_max_size:
                oldest_key = next(iter(llm_cache))
                del llm_cache[oldest_key]
            
            llm_cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {"extracted_attributes": {}, "search_combinations": []}

def generate_search_combinations(attributes: dict) -> list:
    """Generate all meaningful combinations of attributes for comprehensive search"""
    if not attributes:
        return []
    
    combinations_list = []
    attr_keys = list(attributes.keys())
    attr_items = list(attributes.items())
    
    # Full combination (all attributes)
    if len(attr_items) > 1:
        combinations_list.append({
            "combination_type": "full",
            "filters": dict(attr_items)
        })
    
    # Triple combinations
    if len(attr_items) >= 3:
        for combo in combinations(attr_items, 3):
            combinations_list.append({
                "combination_type": "triple", 
                "filters": dict(combo)
            })
    
    # Pair combinations
    if len(attr_items) >= 2:
        for combo in combinations(attr_items, 2):
            combinations_list.append({
                "combination_type": "pair",
                "filters": dict(combo)
            })
    
    # Individual attributes
    for key, value in attr_items:
        combinations_list.append({
            "combination_type": "individual",
            "filters": {key: value}
        })
    
    return combinations_list

def generate_all_combinations(attributes: dict) -> list:
    """Generate ALL possible combinations with UX-optimized priority ordering"""
    if not attributes:
        return []
    
    all_combinations = []
    attr_items = list(attributes.items())
    
    print(f"Generating ALL combinations for attributes: {list(attributes.keys())}")
    
    # Priority 1: Full combination (all attributes) - most specific, best results
    if len(attr_items) >= 2:
        combo_dict = dict(attr_items)
        all_combinations.append({
            "combination_type": "full",
            "filters": combo_dict,
            "priority": 1
        })
    
    # Priority 2: Triple combinations (3 attributes) - high specificity
    if len(attr_items) >= 3:
        for combo in combinations(attr_items, 3):
            combo_dict = dict(combo)
            all_combinations.append({
                "combination_type": "triple",
                "filters": combo_dict,
                "priority": 2
            })
    
    # Priority 3: Pair combinations (2 attributes) - medium specificity  
    if len(attr_items) >= 2:
        for combo in combinations(attr_items, 2):
            combo_dict = dict(combo)
            all_combinations.append({
                "combination_type": "pair", 
                "filters": combo_dict,
                "priority": 3
            })
    
    # Priority 4: Individual attributes - broadest search, fallback
    for item in attr_items:
        combo_dict = dict([item])
        all_combinations.append({
            "combination_type": "individual",
            "filters": combo_dict,
            "priority": 4
        })
    
    print(f"Generated {len(all_combinations)} total combinations (prioritized for UX):")
    for combo in all_combinations:
        filters = combo["filters"]
        print(f"  Priority {combo['priority']} - {combo['combination_type']}: {', '.join(f'{k}={v}' for k, v in filters.items())}")
    
    return all_combinations

def get_clothing_recommendations_by_text(user_description: str, limit: int = 10) -> Dict[str, Any]:
    """Get clothing recommendations using prioritized search strategy: highest to lowest feature count"""
    
    # Extract attributes and get search combinations from LLM
    llm_result = extract_attributes_with_gemini(user_description)
    
    extracted_attributes = llm_result.get("extracted_attributes", {})
    llm_combinations = llm_result.get("search_combinations", [])
    
    # Enhanced fallback: ensure ALL possible combinations are generated
    if not llm_combinations and extracted_attributes:
        llm_combinations = generate_search_combinations(extracted_attributes)
    
    # ENHANCEMENT: Force generation of ALL combinations if LLM missed some
    if extracted_attributes:
        complete_combinations = generate_all_combinations(extracted_attributes)
        # Merge with LLM combinations, avoiding duplicates
        existing_combos = {str(sorted(c.get("filters", {}).items())) for c in llm_combinations}
        for combo in complete_combinations:
            combo_key = str(sorted(combo.get("filters", {}).items()))
            if combo_key not in existing_combos:
                llm_combinations.append(combo)
    
    print(f"Extracted attributes: {extracted_attributes}")
    print(f"Total search combinations: {len(llm_combinations)}")
    
    # PRIORITIZED SEARCH STRATEGY: highest to lowest feature count
    all_results = []
    seen_ids = set()
    
    # Sort combinations by feature count (descending): highest to lowest
    def get_feature_count(combo):
        return len(combo.get("filters", {}))
    
    llm_combinations.sort(key=get_feature_count, reverse=True)
    
    combinations_tried = []
    
    # Execute searches in priority order: highest to lowest feature count
    for i, combo in enumerate(llm_combinations):
        filters = combo.get("filters", {})
        combo_type = combo.get("combination_type", "unknown")
        feature_count = len(filters)
        
        if not filters:
            continue
            
        print(f"Priority search {i+1}/{len(llm_combinations)}: {feature_count} features - {combo_type} - {filters}")
        combinations_tried.append(f"{combo_type}({feature_count} features)")
        
        # Search with smaller batches for faster initial results
        search_limit = min(15, limit * 2)
        search_results = search_by_metadata(filters=filters, limit=search_limit)
        
        # Process results and add new ones
        batch_added = 0
        for result in search_results:
            if len(all_results) >= limit * 2:  # Allow some overflow for variety
                break
                
            original_id = result.payload.get('original_id', 'N/A')
            if original_id != 'N/A' and original_id not in seen_ids:
                seen_ids.add(original_id)
                
                # Create clothing detail with combination info
                image_path = os.path.join(CLOTHES_PATH, f"{original_id}.jpg")
                if os.path.exists(image_path):
                    all_results.append({
                        "id": len(all_results) + 1,
                        "image_id": original_id,
                        "name": f"Fashion Item {original_id}",
                        "category": result.payload.get('clothing_type', 'Fashion Item'),
                        "image": f"http://localhost:8001/static/clothes/{original_id}.jpg",
                        "combination_type": combo_type,
                        "matching_filters": filters,
                        "reason": f"Found via {combo_type}: {', '.join(f'{k}={v}' for k, v in filters.items())}",
                        "search_order": i + 1
                    })
                    batch_added += 1
        
        print(f"  → Found {batch_added} new items (total: {len(all_results)})")
        
        # Continue searching all combinations for comprehensive coverage
        # No early termination to ensure all combinations are tried
    
    # Generate detailed strategy description
    strategy_description = f"Progressive search: {len(combinations_tried)} combinations tried - {' → '.join(combinations_tried[:5])}{'...' if len(combinations_tried) > 5 else ''}"
    
    print(f"Final result: {len(all_results)} unique recommendations from {len(combinations_tried)} combinations")
    
    return {
        "recommendations": all_results[:limit],
        "total_found": len(all_results),
        "strategy": strategy_description,
        "combinations_tried": combinations_tried,
        "extracted_attributes": extracted_attributes,
        "search_coverage": f"{len(combinations_tried)} of {len(llm_combinations)} combinations",
        "error": None
    }

@app.get("/")
async def root():
    return {"message": "Fashion Recommender API is running"}

@app.post("/recommend")
async def recommend_clothing(file: UploadFile = File(...), limit: int = 10):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate limit parameter
    if limit not in [5, 10, 15, 20]:
        raise HTTPException(status_code=400, detail="Limit must be one of: 5, 10, 15, 20")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        result = get_clothing_recommendations_by_uploaded_image(image, limit)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}

@app.post("/search")
async def search_clothing(request: SearchRequest):
    """AI-powered clothing search endpoint with comprehensive combination coverage"""
    try:
        print(f"Search request received: {request.query}, limit: {request.limit}")
        
        # Get recommendations using comprehensive search strategy
        result = get_clothing_recommendations_by_text(
            user_description=request.query,
            limit=request.limit
        )
        
        return result
        
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        return {
            "recommendations": [],
            "total_found": 0,
            "strategy": "Error occurred during search",
            "combinations_tried": [],
            "extracted_attributes": {},
            "search_coverage": "0 of 0 combinations",
            "error": str(e)
        }

@app.post("/search/progressive")
async def search_clothing_progressive(request: SearchRequest):
    """Progressive search that returns initial results quickly, then continues searching"""
    try:
        print(f"Progressive search request: {request.query}, limit: {request.limit}")
        
        # Extract attributes first
        llm_result = extract_attributes_with_gemini(request.query)
        extracted_attributes = llm_result.get("extracted_attributes", {})
        
        if not extracted_attributes:
            return {
                "recommendations": [],
                "total_found": 0,
                "strategy": "No attributes extracted",
                "combinations_tried": [],
                "extracted_attributes": {},
                "search_coverage": "0 of 0 combinations",
                "error": "Could not extract any fashion attributes from the query"
            }
        
        # Generate ALL possible combinations
        all_combinations = generate_all_combinations(extracted_attributes)
        
        # Sort for progressive display (individual first, then pairs, etc.)
        combo_priority = {"individual": 0, "pair": 1, "triple": 2, "full": 3}
        all_combinations.sort(key=lambda x: combo_priority.get(x.get("combination_type", "individual"), 0))
        
        # Progressive search - return quick results
        quick_results = []
        seen_ids = set()
        combinations_tried = []
        
        print(f"Starting progressive search with {len(all_combinations)} combinations")
        
        # First, get individual attribute results (fastest)
        for combo in all_combinations:
            if combo.get("combination_type") != "individual":
                continue
                
            filters = combo.get("filters", {})
            combo_type = combo.get("combination_type", "unknown")
            
            if not filters:
                continue
                
            print(f"Quick search: {combo_type} - {filters}")
            combinations_tried.append(f"{combo_type}({', '.join(filters.keys())})")
            
            search_results = search_by_metadata(filters=filters, limit=10)
            
            for result in search_results:
                if len(quick_results) >= request.limit:
                    break
                    
                original_id = result.payload.get('original_id', 'N/A')
                if original_id != 'N/A' and original_id not in seen_ids:
                    seen_ids.add(original_id)
                    
                    image_path = os.path.join(CLOTHES_PATH, f"{original_id}.jpg")
                    if os.path.exists(image_path):
                        quick_results.append({
                            "id": len(quick_results) + 1,
                            "image_id": original_id,
                            "name": f"Fashion Item {original_id}",
                            "category": result.payload.get('clothing_type', 'Fashion Item'),
                            "image": f"http://localhost:8001/static/clothes/{original_id}.jpg",
                            "combination_type": combo_type,
                            "matching_filters": filters,
                            "reason": f"Quick match: {', '.join(f'{k}={v}' for k, v in filters.items())}",
                            "search_order": len(combinations_tried)
                        })
            
            if len(quick_results) >= request.limit:
                break
        
        # Return quick results with metadata about continuing search
        return {
            "recommendations": quick_results,
            "total_found": len(quick_results),
            "strategy": f"Progressive search: Quick results from individual attributes, {len(all_combinations) - len(combinations_tried)} combinations remaining",
            "combinations_tried": combinations_tried,
            "extracted_attributes": extracted_attributes,
            "search_coverage": f"{len(combinations_tried)} of {len(all_combinations)} combinations (quick mode)",
            "is_progressive": True,
            "remaining_combinations": len(all_combinations) - len(combinations_tried),
            "error": None
        }
        
    except Exception as e:
        print(f"Error in progressive search endpoint: {e}")
        return {
            "recommendations": [],
            "total_found": 0,
            "strategy": "Error in progressive search",
            "combinations_tried": [],
            "extracted_attributes": {},
            "search_coverage": "0 of 0 combinations",
            "is_progressive": False,
            "remaining_combinations": 0,
            "error": str(e)
        }

@app.post("/search/continuous")
async def search_continuous_endpoint(request: SearchRequest):
    """Continuous search that provides more results progressively in priority order by feature count"""
    try:
        user_description = request.query
        limit = request.limit if request.limit else 10
        offset = getattr(request, 'offset', 0)  # Starting position for pagination
        
        print(f"\n=== CONTINUOUS SEARCH REQUEST ===")
        print(f"Query: {user_description}")
        print(f"Limit: {limit}")
        print(f"Offset: {offset}")
        
        # Extract attributes (use cache if available)
        llm_result = extract_attributes_with_gemini(user_description)
        extracted_attributes = llm_result.get("extracted_attributes", {})
        
        if not extracted_attributes:
            return {
                "recommendations": [],
                "total_found": 0,
                "strategy": "No attributes found for continuous search",
                "combinations_tried": [],
                "extracted_attributes": {},
                "search_coverage": "0 of 0 combinations",
                "has_more": False,
                "next_offset": offset,
                "error": "Could not extract attributes"
            }
        
        # Generate ALL combinations with priority ordering (highest to lowest feature count)
        all_combinations = generate_all_combinations(extracted_attributes)
        
        if not all_combinations:
            return {
                "recommendations": [],
                "total_found": 0,
                "strategy": "No combinations generated",
                "combinations_tried": [],
                "extracted_attributes": extracted_attributes,
                "search_coverage": "0 of 0 combinations",
                "has_more": False,
                "next_offset": offset,
                "error": None
            }
        
        # Sort by feature count (highest to lowest)
        def get_feature_count(combo):
            return len(combo.get("filters", {}))
        
        all_combinations.sort(key=get_feature_count, reverse=True)
        
        # Collect ALL results from ALL combinations first
        all_possible_results = []
        combinations_tried = []
        seen_ids = set()
        
        print(f"Collecting results from {len(all_combinations)} combinations in priority order")
        
        for combo in all_combinations:
            filters = combo.get("filters", {})
            combination_type = combo.get("combination_type", "unknown")
            feature_count = len(filters)
            
            print(f"Searching {feature_count} features - {combination_type}: {filters}")
            
            # Search with this combination
            results = search_by_metadata(filters=filters, limit=50)  # Get more results per combination
            
            combinations_tried.append(f"{combination_type}({feature_count} features)")
            
            if results:
                for record in results:
                    original_id = record.payload.get('original_id', 'N/A')
                    
                    if original_id != 'N/A' and original_id and original_id not in seen_ids:
                        seen_ids.add(original_id)
                        
                        result_dict = {
                            "id": len(all_possible_results) + 1,
                            "image_id": original_id,
                            "name": f"Fashion Item {original_id}",
                            "category": record.payload.get('clothing_type', 'Fashion Item'),
                            "image": f"http://localhost:8001/static/clothes/{original_id}.jpg",
                            "combination_type": combination_type,
                            "matching_filters": filters,
                            "reason": f"Found via {combination_type} search ({feature_count} features): {', '.join(f'{k}={v}' for k, v in filters.items())}",
                            "feature_count": feature_count,
                            "search_order": len(all_possible_results)
                        }
                        
                        all_possible_results.append(result_dict)
                        print(f"Added result: {result_dict['name']} ({feature_count} features)")
        
        print(f"Total collected results: {len(all_possible_results)}")
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        paginated_results = all_possible_results[start_idx:end_idx]
        has_more = end_idx < len(all_possible_results)
        next_offset = end_idx if has_more else len(all_possible_results)
        
        print(f"Returning results {start_idx}-{end_idx-1} of {len(all_possible_results)}")
        print(f"Has more: {has_more}, Next offset: {next_offset}")
        
        return {
            "recommendations": paginated_results,
            "total_found": len(paginated_results),
            "total_available": len(all_possible_results),
            "strategy": f"Continuous search: Showing results {start_idx+1}-{start_idx+len(paginated_results)} of {len(all_possible_results)} (priority order by feature count)",
            "combinations_tried": combinations_tried,
            "extracted_attributes": extracted_attributes,
            "search_coverage": f"{len(combinations_tried)} combinations searched",
            "has_more": has_more,
            "next_offset": next_offset,
            "feature_breakdown": {
                "4_features": len([r for r in all_possible_results if r["feature_count"] == 4]),
                "3_features": len([r for r in all_possible_results if r["feature_count"] == 3]),
                "2_features": len([r for r in all_possible_results if r["feature_count"] == 2]),
                "1_feature": len([r for r in all_possible_results if r["feature_count"] == 1])
            },
            "error": None
        }
        
    except Exception as e:
        print(f"Error in continuous search endpoint: {e}")
        return {
            "recommendations": [],
            "total_found": 0,
            "strategy": "Error in continuous search",
            "combinations_tried": [],
            "extracted_attributes": {},
            "search_coverage": "0 of 0 combinations",
            "has_more": False,
            "next_offset": offset,
            "error": str(e)
        }

@app.post("/search/more")
async def search_more_endpoint(request: SearchRequest):
    """Expand search with remaining combinations for better coverage"""
    try:
        user_description = request.query
        limit = request.limit if request.limit else 10
        
        print(f"\n=== SEARCH MORE REQUEST ===")
        print(f"Query: {user_description}")
        print(f"Limit: {limit}")
        
        # Extract attributes (use cache if available)
        llm_result = extract_attributes_with_gemini(user_description)
        extracted_attributes = llm_result.get("extracted_attributes", {})
        
        if not extracted_attributes:
            return {
                "recommendations": [],
                "total_found": 0,
                "strategy": "No attributes found to expand search",
                "combinations_tried": [],
                "extracted_attributes": {},
                "search_coverage": "0 of 0 combinations",
                "is_expanded": True,
                "error": "Could not extract attributes for expansion"
            }
        
        # Generate ALL combinations with priority ordering
        all_combinations = generate_all_combinations(extracted_attributes)
        
        if not all_combinations:
            return {
                "recommendations": [],
                "total_found": 0,
                "strategy": "No combinations generated for expanded search",
                "combinations_tried": [],
                "extracted_attributes": extracted_attributes,
                "search_coverage": "0 of 0 combinations",
                "is_expanded": True,
                "error": None
            }
        
        # For "Search More", skip the first few high-priority combinations 
        # and focus on broader searches (pairs and individuals)
        expanded_results = []
        combinations_tried = []
        
        # Start from priority 3 (pairs) and 4 (individuals) for expansion
        expansion_combinations = [combo for combo in all_combinations if combo.get("priority", 4) >= 3]
        
        print(f"Expanding search with {len(expansion_combinations)} combinations (pairs + individuals)")
        
        for combo in expansion_combinations:
            if len(expanded_results) >= limit:
                break
                
            filters = combo["filters"]
            combination_type = combo["combination_type"]
            
            print(f"Trying expanded combination: {combination_type} - {filters}")
            
            # Search with this combination
            results = search_by_metadata(filters=filters, limit=limit)
            
            print(f"Search returned {len(results)} results for {combination_type} - {filters}")
            
            combinations_tried.append(f"{combination_type}: {', '.join(f'{k}={v}' for k, v in filters.items())}")
            
            if results:
                for record in results:
                    if len(expanded_results) >= limit:
                        break
                    
                    # Convert QDrant record to dictionary
                    original_id = record.payload.get('original_id', 'N/A')
                    print(f"Processing record with original_id: {original_id}")
                    
                    if original_id != 'N/A' and original_id:
                        result_dict = {
                            "id": len(expanded_results) + 1,
                            "image_id": original_id,
                            "name": f"Fashion Item {original_id}",
                            "category": record.payload.get('clothing_type', 'Fashion Item'),
                            "image": f"http://localhost:8001/static/clothes/{original_id}.jpg",
                            "combination_type": combination_type,
                            "matching_filters": filters,
                            "reason": f"Found via {combination_type} search: {', '.join(f'{k}={v}' for k, v in filters.items())}"
                        }
                        
                        # Avoid duplicates
                        if not any(existing["image_id"] == result_dict["image_id"] for existing in expanded_results):
                            expanded_results.append(result_dict)
                            print(f"Added result: {result_dict['name']} (ID: {result_dict['image_id']})")
                        else:
                            print(f"Skipped duplicate: {original_id}")
        
        print(f"Expanded search completed: {len(expanded_results)} results from {len(combinations_tried)} combinations")
        
        return {
            "recommendations": expanded_results,
            "total_found": len(expanded_results),
            "strategy": f"Expanded search: Found {len(expanded_results)} additional items using broader combinations (pairs + individuals)",
            "combinations_tried": combinations_tried,
            "extracted_attributes": extracted_attributes,
            "search_coverage": f"{len(combinations_tried)} of {len(expansion_combinations)} expansion combinations",
            "is_expanded": True,
            "error": None
        }
        
    except Exception as e:
        print(f"Error in search more endpoint: {e}")
        return {
            "recommendations": [],
            "total_found": 0,
            "strategy": "Error in expanded search",
            "combinations_tried": [],
            "extracted_attributes": {},
            "search_coverage": "0 of 0 combinations",
            "is_expanded": True,
            "error": str(e)
        }

@app.get("/random-items")
async def get_random_items(limit: int = 10):
    """Get random fashion items from the database for catalogue display"""
    try:
        # Scroll through the collection to get random items
        import random
        
        # First, get the total count of items in the collection
        collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
        total_points = collection_info.points_count
        
        if total_points == 0:
            return {
                "items": [],
                "total_count": 0,
                "error": "No items found in database"
            }
        
        # Generate random offsets to get diverse items
        random_offsets = random.sample(range(0, min(total_points, 1000)), min(limit, total_points))
        
        random_items = []
        
        for offset in random_offsets:
            # Scroll to get items at random positions
            points, _ = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=1,
                offset=offset,
                with_payload=True
            )
            
            if points:
                point = points[0]
                item = {
                    "id": point.payload.get("original_id", "unknown"),
                    "name": f"Fashion Item {point.payload.get('original_id', 'Unknown')}",
                    "category": point.payload.get("clothing_type", "Unknown"),
                    "image": f"http://localhost:8001/static/clothes/{point.payload.get('original_id', 'unknown')}.jpg",
                    "dominant_color": point.payload.get("dominant_color", "Unknown"),
                    "sleeve_type": point.payload.get("sleeve_type", "Unknown"),
                    "pattern_type": point.payload.get("pattern_type", "Unknown"),
                    "description": f"{point.payload.get('clothing_type', 'Fashion item')} with {point.payload.get('pattern_type', 'classic')} pattern"
                }
                random_items.append(item)
                
                if len(random_items) >= limit:
                    break
        
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

def main():
    setup_qdrant_collection()
    print("FastAPI server ready. Use uvicorn to run the server.")

if __name__ == "__main__":
    import uvicorn
    print("Starting Fashion Recommender API on port 8001...")
    print("Face color analysis and clothing recommendations")
    print("Static files: http://localhost:8001/static/clothes/")
    print("API docs: http://localhost:8001/docs")
    print("Features: Face analysis, color palette detection, visual search")
    uvicorn.run(app, host="0.0.0.0", port=8001)
