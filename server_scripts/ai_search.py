import os
import json
import uuid
import numpy as np
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient, models as qdrant_models
import google.generativeai as genai
from ast import literal_eval
from itertools import combinations

QDRANT_URL = "https://c977d41b-092f-4746-8055-e0c1974ed673.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-IpipKy3ymQr1qZIAvAzNZ87K1e0Qxd8B2MNKTWY07w"
QDRANT_COLLECTION_NAME = "fashion_clip_recommender"
VECTOR_SIZE = 2048
NAMESPACE_UUID = uuid.NAMESPACE_DNS

# Get the parent directory for data files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOTHES_PATH = os.path.join(BASE_DIR, "clothes_tryon_dataset", "test", "cloth")

# Pydantic models for API
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    offset: int = 0  # For pagination

class SearchResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    total_found: int
    extracted_attributes: Dict[str, Any]
    search_combinations: List[Dict[str, Any]]
    has_more: bool = False
    next_offset: int = 0
    error: str | None = None

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
genai.configure(api_key="AIzaSyCMBWqTE4bIKlG8klXWrIHW4_WrjBIbEyU")

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

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_qdrant_collection()
    print("AI Fashion Search API started successfully!")
    yield
    # Shutdown (if needed)
    print("AI Fashion Search API shutting down...")

# FastAPI app setup
app = FastAPI(
    title="AI Fashion Search API", 
    description="Advanced fashion search with AI attribute extraction",
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
from fastapi.staticfiles import StaticFiles
app.mount("/static/clothes", StaticFiles(directory=CLOTHES_PATH), name="clothes")

def search_by_metadata(filters: dict, limit: int = 10) -> list:
    must_conditions = []
    for key, value in filters.items():
        if value is None or str(value).strip() == '' or str(value).lower() == 'n/a':
            continue

        if isinstance(value, list):
            # Handle list values with MatchAny
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchAny(any=value)
                )
            )
        elif key in ["gender_suitability", "occasion_suitability"]:
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

def extract_attributes_with_gemini(user_prompt: str) -> dict:
    llm_instruction = f"""
    You are an intelligent fashion search assistant. Your task is to extract clothing attributes from user descriptions and create comprehensive search combinations for maximum search coverage.

    TASK 1 - ATTRIBUTE EXTRACTION:
    Extract the following attributes from the user's description:

    CLOTHING ATTRIBUTES:
    - clothing_type: [Blouse, Shirt, T-shirt, Hoodie, Sweater, Blazer, Jacket, Dress, Skirt, Pants, Shorts, Jeans, Coat, Vest, Tank Top, Cardigan, Jumpsuit, Romper, Leggings, Tunic]
    - sleeve_type: [Sleeveless, Long Sleeve, Short Sleeve, Three-Quarter Sleeve, Cap Sleeve, Bell Sleeve, Puff Sleeve]
    - neckline: [Collar, Crew Neck, V-Neck, High Neck, Hooded, Scoop Neck, Strapless, Off-Shoulder, Boat Neck, Cowl Neck, Halter, Square Neck]
    - closure_type: [None, Zipper, Tie, Button, Velcro, Pullover, Wrap, Drawstring, Snap, Hook, Lace-up]
    - pattern_type: [Striped, Graphic Print, Solid, Floral, Polka Dot, Animal Print, Checked, Plaid, Geometric, Abstract, Embroidered, Paisley, Tie-Dye, Leopard, Zebra]
    - gender_suitability: [Male, Female, Unisex, Child]
    - occasion_suitability: [Casual, Formal, Party Wear, Sportswear, Business Casual, Beachwear, Lounge Wear, Work Wear, Evening Wear, Wedding]
    - dominant_color: [Red, Blue, Green, Yellow, Black, White, Brown, Gray, Pink, Purple, Orange, Teal, Mint, Aqua, Gold, Peach, Burgundy, Lavender, Navy, Beige, Coral, Turquoise, Maroon, Olive, Cream]
    - secondary_color: [Red, Blue, Green, Yellow, Black, White, Brown, Gray, Pink, Purple, Orange, Teal, Mint, Aqua, Gold, Peach, Burgundy, Lavender, Navy, Beige, Coral, Turquoise, Maroon, Olive, Cream]

    TASK 2 - INTELLIGENT SEARCH COMBINATIONS:
    Create multiple search filter combinations to maximize search coverage:

    1. INDIVIDUAL ATTRIBUTES: Each attribute searched separately
    2. PAIRED COMBINATIONS: Two attributes combined (e.g., red + dress, polka dot + dress, red + polka dot)
    3. TRIPLE COMBINATIONS: Three attributes combined when applicable
    4. FULL COMBINATION: All extracted attributes together

    EXAMPLE FOR "red dress with polka dots":
    - Individual: {{"dominant_color": "Red"}}, {{"clothing_type": "Dress"}}, {{"pattern_type": "Polka Dot"}}
    - Pairs: {{"dominant_color": "Red", "clothing_type": "Dress"}}, {{"dominant_color": "Red", "pattern_type": "Polka Dot"}}, {{"clothing_type": "Dress", "pattern_type": "Polka Dot"}}
    - Full: {{"dominant_color": "Red", "clothing_type": "Dress", "pattern_type": "Polka Dot"}}

    RULES:
    - Only include attributes explicitly mentioned in the user input
    - Use exact values from the provided attribute lists
    - For colors, always use the closest match from the color list
    - Generate ALL meaningful combinations (individual, paired, triple, full)
    - Order combinations from most specific (full) to least specific (individual)

    USER INPUT: {user_prompt}

    RESPOND with a JSON object containing:
    {{
        "extracted_attributes": {{"attribute": "value", ...}},
        "search_combinations": [
            {{"combination_type": "full", "filters": {{"attr1": "val1", "attr2": "val2", ...}}}},
            {{"combination_type": "triple", "filters": {{"attr1": "val1", "attr2": "val2", "attr3": "val3"}}}},
            {{"combination_type": "pair", "filters": {{"attr1": "val1", "attr2": "val2"}}}},
            {{"combination_type": "individual", "filters": {{"attr1": "val1"}}}}
        ]
    }}

    Respond ONLY with valid JSON. No markdown, no code blocks, no explanations.
    """

    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content(llm_instruction)

    text_out = response.text.strip()
    print(f"LLM Response: {text_out}")

    # Clean response
    if text_out.startswith('```'):
        text_out = '\n'.join([line for line in text_out.splitlines() if not line.strip().startswith('```')])

    try:
        result = json.loads(text_out)
        return result
    except json.JSONDecodeError:
        try:
            result = literal_eval(text_out)
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

def get_clothing_recommendations_by_text(user_description: str, limit: int = 10) -> dict:
    """Get clothing recommendations using comprehensive search strategy"""
    
    # Extract attributes and get search combinations from LLM
    llm_result = extract_attributes_with_gemini(user_description)
    
    extracted_attributes = llm_result.get("extracted_attributes", {})
    llm_combinations = llm_result.get("search_combinations", [])
    
    # Fallback: generate combinations if LLM didn't provide them
    if not llm_combinations and extracted_attributes:
        llm_combinations = generate_search_combinations(extracted_attributes)
    
    print(f"Extracted attributes: {extracted_attributes}")
    print(f"Search combinations: {len(llm_combinations)}")
    
    all_results = []
    seen_ids = set()
    combinations_tried = []
    
    # Execute searches for each combination
    for combo in llm_combinations:
        filters = combo.get("filters", {})
        combo_type = combo.get("combination_type", "unknown")
        
        if not filters:
            continue
            
        print(f"Searching with {combo_type}: {filters}")
        combinations_tried.append(f"{combo_type}: {filters}")
        
        search_results = search_by_metadata(filters=filters, limit=limit)
        
        # Extract unique image IDs and format for API response
        for result in search_results:
            original_id = result.payload.get('original_id', 'N/A')
            if original_id != 'N/A' and original_id not in seen_ids:
                seen_ids.add(original_id)
                all_results.append({
                    "id": len(all_results) + 1,
                    "image_id": original_id,
                    "name": f"Fashion Item {original_id}",
                    "category": result.payload.get('clothing_type', 'Fashion Item'),
                    "image": f"http://localhost:8002/static/clothes/{original_id}.jpg",
                    "combination_type": combo_type,
                    "matching_filters": filters,
                    "reason": f"Found via {combo_type} search: {', '.join(f'{k}={v}' for k, v in filters.items())}"
                })
        
        # Stop if we have enough results
        if len(all_results) >= limit:
            break
    
    print(f"Found {len(all_results)} unique recommendations")
    
    return {
        "recommendations": all_results[:limit],
        "total_found": len(all_results),
        "extracted_attributes": extracted_attributes,
        "search_combinations": combinations_tried,
        "error": None
    }

# FastAPI endpoints
@app.get("/")
async def root():
    return {"message": "AI Fashion Search API is running", "endpoints": ["/search", "/health", "/docs"]}

@app.get("/health")
async def health_check():
    try:
        # Test Qdrant connection
        collections = client.get_collections()
        return {
            "status": "healthy",
            "qdrant_connection": "ok",
            "collections": len(collections.collections)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "qdrant_connection": f"error: {e}",
            "collections": 0
        }

@app.post("/search", response_model=SearchResponse)
async def search_clothing(request: SearchRequest):
    """AI-powered clothing search with attribute extraction"""
    try:
        print(f"AI Search request: {request.query}, limit: {request.limit}")
        
        result = get_clothing_recommendations_by_text(
            user_description=request.query,
            limit=request.limit
        )
        
        return SearchResponse(**result)
        
    except Exception as e:
        print(f"Error in AI search endpoint: {e}")
        return SearchResponse(
            recommendations=[],
            total_found=0,
            extracted_attributes={},
            search_combinations=[],
            error=str(e)
        )

@app.post("/search/continuous")
async def search_continuous_endpoint(request: SearchRequest):
    """Continuous search that provides more results progressively"""
    try:
        user_description = request.query
        limit = request.limit if request.limit else 10
        offset = getattr(request, 'offset', 0)
        
        print(f"\n=== AI CONTINUOUS SEARCH REQUEST ===")
        print(f"Query: {user_description}")
        print(f"Limit: {limit}")
        print(f"Offset: {offset}")
        
        # Get comprehensive results
        result = get_clothing_recommendations_by_text(user_description, limit=limit * 2)
        all_recommendations = result.get("recommendations", [])
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        paginated_results = all_recommendations[start_idx:end_idx]
        has_more = end_idx < len(all_recommendations)
        next_offset = end_idx if has_more else len(all_recommendations)
        
        return {
            "recommendations": paginated_results,
            "total_found": len(paginated_results),
            "total_available": len(all_recommendations),
            "strategy": f"AI-powered continuous search: Showing results {start_idx+1}-{start_idx+len(paginated_results)} of {len(all_recommendations)}",
            "combinations_tried": result.get("search_combinations", []),
            "extracted_attributes": result.get("extracted_attributes", {}),
            "search_coverage": f"AI attribute extraction with {len(result.get('search_combinations', []))} combinations",
            "has_more": has_more,
            "next_offset": next_offset,
            "error": None
        }
        
    except Exception as e:
        print(f"Error in AI continuous search endpoint: {e}")
        return {
            "recommendations": [],
            "total_found": 0,
            "strategy": "Error in AI continuous search",
            "combinations_tried": [],
            "extracted_attributes": {},
            "search_coverage": "0 combinations",
            "has_more": False,
            "next_offset": offset,
            "error": str(e)
        }

@app.post("/search/progressive")
async def search_progressive_endpoint(request: SearchRequest):
    """Progressive search that returns initial results quickly"""
    try:
        print(f"AI Progressive search request: {request.query}, limit: {request.limit}")
        
        # Quick search with basic attributes first
        result = get_clothing_recommendations_by_text(
            user_description=request.query,
            limit=request.limit
        )
        
        return {
            "recommendations": result.get("recommendations", []),
            "total_found": len(result.get("recommendations", [])),
            "strategy": f"AI progressive search: Quick results with attribute extraction",
            "combinations_tried": result.get("search_combinations", []),
            "extracted_attributes": result.get("extracted_attributes", {}),
            "search_coverage": f"AI-powered with {len(result.get('search_combinations', []))} combinations",
            "is_progressive": True,
            "remaining_combinations": 0,
            "error": None
        }
        
    except Exception as e:
        print(f"Error in AI progressive search endpoint: {e}")
        return {
            "recommendations": [],
            "total_found": 0,
            "strategy": "Error in AI progressive search",
            "combinations_tried": [],
            "extracted_attributes": {},
            "search_coverage": "0 combinations",
            "is_progressive": False,
            "remaining_combinations": 0,
            "error": str(e)
        }

@app.post("/search/more")
async def search_more_endpoint(request: SearchRequest):
    """Expand search with broader combinations"""
    try:
        print(f"AI Search More request: {request.query}, limit: {request.limit}")
        
        # Get expanded results with more combinations
        result = get_clothing_recommendations_by_text(
            user_description=request.query,
            limit=request.limit * 2  # Get more results for expansion
        )
        
        return {
            "recommendations": result.get("recommendations", []),
            "total_found": len(result.get("recommendations", [])),
            "strategy": f"AI expanded search: Broader attribute combinations",
            "combinations_tried": result.get("search_combinations", []),
            "extracted_attributes": result.get("extracted_attributes", {}),
            "search_coverage": f"AI-expanded with {len(result.get('search_combinations', []))} combinations",
            "is_expanded": True,
            "error": None
        }
        
    except Exception as e:
        print(f"Error in AI search more endpoint: {e}")
        return {
            "recommendations": [],
            "total_found": 0,
            "strategy": "Error in AI expanded search",
            "combinations_tried": [],
            "extracted_attributes": {},
            "search_coverage": "0 combinations",
            "is_expanded": True,
            "error": str(e)
        }

@app.get("/random-items")
async def get_random_items(limit: int = 10):
    """Get random fashion items from the database"""
    try:
        import random
        
        # Get a sample of items from Qdrant
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
                "image": f"http://localhost:8002/static/clothes/{point.payload.get('original_id', 'unknown')}.jpg",
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

@app.get("/test-search")
async def test_search():
    """Test endpoint with predefined search"""
    try:
        test_query = "red dress"
        result = get_clothing_recommendations_by_text(test_query, limit=5)
        return {
            "test_query": test_query,
            "result": result
        }
    except Exception as e:
        return {
            "test_query": "red dress",
            "error": str(e)
        }

if __name__ == "__main__":
    import sys
    import uvicorn
    
    # Always run as FastAPI server
    print("Starting AI Fashion Search API server on port 8002...")
    print("AI-powered search with attribute extraction")
    print("Static files: http://localhost:8002/static/clothes/")
    print("API docs: http://localhost:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8002)
