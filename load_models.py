import os
import torch
import requests
import pickle
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux.open_pose import OpenposeDetector
from segment_anything import sam_model_registry, SamPredictor

def check_existing_models():
    """Check what models are already available"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    loaded_models_dir = os.path.join(base_dir, "loaded_models")
    
    existing_models = {}
    
    # Check for style_transfer_models.pkl
    style_models_path = os.path.join(loaded_models_dir, "style_transfer_models.pkl")
    if os.path.exists(style_models_path):
        try:
            with open(style_models_path, 'rb') as f:
                style_models = pickle.load(f)
            existing_models['style_transfer'] = style_models
            print(f"✅ Found existing style transfer models: {list(style_models.keys())}")
        except Exception as e:
            print(f"❌ Error loading style transfer models: {e}")
    
    # Check for vton_models.pkl
    vton_models_path = os.path.join(loaded_models_dir, "vton_models.pkl")
    if os.path.exists(vton_models_path):
        try:
            with open(vton_models_path, 'rb') as f:
                vton_models = pickle.load(f)
            existing_models['vton'] = vton_models
            print(f"✅ Found existing VTON models: {list(vton_models.keys())}")
        except Exception as e:
            print(f"❌ Error loading VTON models: {e}")
    
    return existing_models

def download_sam_model(device):
    """Download SAM model if not already present"""
    print("📥 Downloading SAM model...")
    sam_model_type = "vit_h"
    sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    # Use loaded_models directory instead of kaggle setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    loaded_models_dir = os.path.join(base_dir, "loaded_models")
    os.makedirs(loaded_models_dir, exist_ok=True)
    
    sam_checkpoint_path = os.path.join(loaded_models_dir, "sam_checkpoint.pth")

    if not os.path.exists(sam_checkpoint_path):
        print(f"⬇️ Downloading SAM checkpoint...")
        response = requests.get(sam_checkpoint_url)
        response.raise_for_status()
        with open(sam_checkpoint_path, "wb") as f:
            f.write(response.content)
        print(f"✅ SAM checkpoint downloaded to: {sam_checkpoint_path}")
    else:
        print(f"✅ SAM checkpoint already exists: {sam_checkpoint_path}")

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    return predictor

def initialize_models(base_model_id, controlnet_model_id, finetuned_ip_adapter_path, device):
    """Initialize and download models only if not already present"""
    print("🔍 Checking existing models...")
    existing_models = check_existing_models()
    
    models_to_load = {}
    
    # Check if we already have the required models
    if 'vton' in existing_models:
        vton_models = existing_models['vton']
        if all(key in vton_models for key in ['pipe', 'predictor', 'openpose_detector']):
            print("✅ All VTON models already loaded!")
            return vton_models['pipe'], vton_models['predictor'], vton_models['openpose_detector']
    
    print("📦 Loading required models...")
    
    # Download SAM model
    predictor = download_sam_model(device)
    models_to_load['predictor'] = predictor
    
    # Load OpenPose detector
    print("📥 Loading OpenPose detector...")
    openpose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    models_to_load['openpose_detector'] = openpose_detector
    
    # Load ControlNet
    print(f"📥 Loading ControlNet: {controlnet_model_id}")
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
    
    # Load Stable Diffusion pipeline
    print(f"📥 Loading Stable Diffusion pipeline: {base_model_id}")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print("📥 Loading IP-Adapter...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

    # Load finetuned IP-Adapter if path provided
    if finetuned_ip_adapter_path and os.path.exists(finetuned_ip_adapter_path):
        finetuned_proj_model_path = os.path.join(finetuned_ip_adapter_path, "try_on_model.bin")
        if os.path.exists(finetuned_proj_model_path):
            print(f"📥 Loading finetuned IP-Adapter: {finetuned_proj_model_path}")
            finetuned_state_dict = torch.load(finetuned_proj_model_path, map_location="cpu")
            ip_adapter_model = pipe.ip_adapters[0]
            ip_adapter_model.image_proj_model.load_state_dict(finetuned_state_dict)
        else:
            print(f"⚠️ Finetuned model not found: {finetuned_proj_model_path}")
    
    models_to_load['pipe'] = pipe
    models_to_load['device'] = device
    
    # Save the models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    loaded_models_dir = os.path.join(base_dir, "loaded_models")
    vton_models_path = os.path.join(loaded_models_dir, "vton_models.pkl")
    
    print(f"💾 Saving VTON models to: {vton_models_path}")
    with open(vton_models_path, 'wb') as f:
        pickle.dump(models_to_load, f)
    
    print("✅ All models loaded and saved successfully!")
    return pipe, predictor, openpose_detector

if __name__ == "__main__":
    # Configuration
    base_model_id = "runwayml/stable-diffusion-inpainting"
    controlnet_model_id = "lllyasviel/sd-controlnet-openpose"
    finetuned_ip_adapter_path = None  # Set this if you have a finetuned model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 Starting model initialization...")
    print(f"Device: {device}")
    
    try:
        pipe, predictor, openpose_detector = initialize_models(
            base_model_id, 
            controlnet_model_id, 
            finetuned_ip_adapter_path, 
            device
        )
        print("🎉 Model initialization completed successfully!")
        print("📁 Models saved in loaded_models/vton_models.pkl")
        
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        import traceback
        traceback.print_exc()