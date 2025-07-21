import os
import json
import uuid
import pickle
import torch
from PIL import Image
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import io
import tempfile
import shutil

class StyleTransferService:
    """Service for handling style transfer operations"""
    
    def __init__(self):
        # Get the parent directory for data files
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(base_dir, "loaded_models")
        self.output_dir = os.path.join(base_dir, "amalgam_outputs")
        self.features_dir = os.path.join(base_dir, "features_extracted")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load models
        self.models = self._load_models()
        self.pipe = self.models['pipe']
        self.canny_detector = self.models['canny_detector']
        self.device = self.models['device']
        
        print("Style Transfer Service initialized successfully!")
    
    def _load_models(self):
        """Load the pretrained models"""
        models_path = os.path.join(self.models_dir, 'style_transfer_models.pkl')
        
        if not os.path.exists(models_path):
            raise FileNotFoundError(
                "Models not found! Please run 'python model_for_gen.py' first to load the models."
            )
        
        print("Loading models from saved file...")
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        
        print("Models loaded successfully!")
        return models
    
    def get_features_from_json(self, image_filename: str) -> Dict[str, Any]:
        """Get features from JSON file for a given image"""
        json_filename = os.path.splitext(image_filename)[0] + '.json'
        json_path = os.path.join(self.features_dir, json_filename)
        
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: JSON file not found for {image_filename}. Using defaults.")
            return {
                'dominant_color': 'white',
                'clothing_type': 'dress',
                'pattern_type': 'solid'
            }
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON for {image_filename}. Using defaults.")
            return {
                'dominant_color': 'white',
                'clothing_type': 'dress', 
                'pattern_type': 'solid'
            }
    
    def generate_style_transfer(self, base_image: Image.Image, style_image: Image.Image, 
                              base_filename: str = "base_image", style_filename: str = "style_image") -> str:
        """
        Generate style transfer between two clothing images
        
        Args:
            base_image: Base image (structure and color source)
            style_image: Style image (pattern source)
            base_filename: Filename for base image (for feature lookup)
            style_filename: Filename for style image (for feature lookup)
        
        Returns:
            Path to the generated image
        """
        print(f"\n Starting Style Transfer ")
        print(f"Base image: {base_filename}")
        print(f"Style image: {style_filename}")
        
        # Get features from JSON files
        base_features = self.get_features_from_json(base_filename)
        style_features = self.get_features_from_json(style_filename)
        
        # Extract attributes
        base_color = base_features.get('dominant_color', 'white')
        base_type = base_features.get('clothing_type', 'dress')
        style_pattern = style_features.get('pattern_type', 'floral print')
        
        print(f"Base color: {base_color}")
        print(f"Base type: {base_type}")
        print(f"Style pattern: {style_pattern}")
        
        # Create control image (Canny edge map)
        print("Creating Canny edge map...")
        control_image = self.canny_detector(base_image, low_threshold=100, high_threshold=200)
        
        # Generate prompt
        prompt = f"a photorealistic {base_type} in shades of {base_color} with a {style_pattern} pattern, best quality, high quality"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
        
        print(f"Prompt: {prompt}")
        
        # Generate image
        print("Generating style transfer...")
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        generated_images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            ip_adapter_image=style_image,
            num_inference_steps=30,
            ip_adapter_scale=0.7,
            generator=generator,
        ).images
        
        # Save result
        final_image = generated_images[0]
        output_filename = f"amalgam_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        final_image.save(output_path)
        
        print(f"Style transfer complete! Saved to: {output_path}")
        return output_path

# Initialize service
try:
    style_service = StyleTransferService()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run 'python model_for_gen.py' first!")
    style_service = None

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if style_service:
        print("Fashion Style Transfer API started successfully!")
    else:
        print("API started but models are not loaded!")
    yield
    # Shutdown (if needed)
    print("Fashion Style Transfer API shutting down...")

# FastAPI app setup
app = FastAPI(
    title="Fashion Style Transfer API",
    description="AI-powered style transfer and image generation",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving generated images
if style_service:
    app.mount("/static/amalgam", StaticFiles(directory=style_service.output_dir), name="amalgam")
else:
    # Fallback path if service not initialized
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "amalgam_outputs")
    os.makedirs(output_dir, exist_ok=True)
    app.mount("/static/amalgam", StaticFiles(directory=output_dir), name="amalgam")

@app.get("/")
async def root():
    return {
        "message": "Fashion Style Transfer API",
        "status": "running" if style_service else "models not loaded",
        "endpoints": {
            "style_transfer": "/styletransfer",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if style_service else "unhealthy",
        "models_loaded": style_service is not None,
        "device": style_service.device if style_service else "unknown"
    }

@app.post("/styletransfer")
async def create_style_transfer(
    base_image: UploadFile = File(..., description="Base clothing image (structure and color source)"),
    style_image: UploadFile = File(..., description="Style clothing image (pattern source)"),
    base_filename: str = Form(default="", description="Original filename of base image for feature lookup"),
    style_filename: str = Form(default="", description="Original filename of style image for feature lookup")
):
    """
    Create a style transfer between two clothing images
    """
    if not style_service:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please run 'python model_for_gen.py' first."
        )
    
    # Validate file types
    if not base_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Base file must be an image")
    
    if not style_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Style file must be an image")
    
    try:
        # Read and process images
        base_contents = await base_image.read()
        style_contents = await style_image.read()
        
        base_pil = Image.open(io.BytesIO(base_contents)).convert('RGB')
        style_pil = Image.open(io.BytesIO(style_contents)).convert('RGB')
        
        # Use provided filenames or generate defaults
        base_fname = base_filename if base_filename else base_image.filename or "base_image.jpg"
        style_fname = style_filename if style_filename else style_image.filename or "style_image.jpg"
        
        # Generate style transfer
        output_path = style_service.generate_style_transfer(
            base_image=base_pil,
            style_image=style_pil,
            base_filename=base_fname,
            style_filename=style_fname
        )
        
        # Get relative path for URL
        output_filename = os.path.basename(output_path)
        result_url = f"/static/amalgam/{output_filename}"
        
        return {
            "success": True,
            "message": "Style transfer completed successfully",
            "result_url": result_url,
            "result_path": output_path,
            "base_image": base_fname,
            "style_image": style_fname
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")

# Alias endpoint for frontend compatibility
@app.post("/style-transfer")
async def create_style_transfer_alias(
    base_image: UploadFile = File(..., description="Base clothing image (structure and color source)"),
    style_image: UploadFile = File(..., description="Style clothing image (pattern source)"),
    base_filename: str = Form(default="", description="Original filename of base image for feature lookup"),
    style_filename: str = Form(default="", description="Original filename of style image for feature lookup")
):
    """
    Create a style transfer between two clothing images (alias endpoint)
    """
    return await create_style_transfer(base_image, style_image, base_filename, style_filename)

@app.get("/download/{filename}")
async def download_result(filename: str):
    """Download a generated result image"""
    if style_service:
        file_path = os.path.join(style_service.output_dir, filename)
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "amalgam_outputs", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='image/png'
    )

@app.get("/listresults")
async def list_results():
    """List all generated result images"""
    if style_service:
        output_dir = style_service.output_dir
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "amalgam_outputs")
    
    if not os.path.exists(output_dir):
        return {"results": []}
    
    files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    results = [{"filename": f, "url": f"/static/amalgam/{f}"} for f in files]
    
    return {"results": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn

    print("Starting Fashion Style Transfer API on port 8003...")
    print("AI-powered style transfer and image generation")
    print("Static files: http://localhost:8003/static/amalgam/")
    print("API docs: http://localhost:8003/docs")
    print("Features: Style transfer, ControlNet, IPAdapter")
    uvicorn.run(app, host="0.0.0.0", port=8003)
