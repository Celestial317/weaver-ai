import os
import torch
import pickle
import warnings
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class ModelLoader:
    """
    One-time model loader for style transfer generation.
    Loads all required models and saves them for reuse.
    """
    
    def __init__(self):
        self.base_model_id = "runwayml/stable-diffusion-v1-5"
        self.controlnet_model_id = "lllyasviel/sd-controlnet-canny"
        self.ip_adapter_model_id = "h94/IP-Adapter"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = "loaded_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        if self.device == "cpu":
            print("Warning: Running on CPU. This will be very slow. Consider using GPU.")
    
    def load_all_models(self):
        """Load all models required for style transfer generation"""
        print("\n--- Loading All Models ---")
        print("This may take several minutes and require ~3-5 GB of disk space...")
        
        try:
            # Load ControlNet
            print("1/4 Loading ControlNet model...")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            print("✅ ControlNet model loaded successfully.")
            
            # Load main pipeline
            print("2/4 Loading Stable Diffusion pipeline...")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.base_model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            ).to(self.device)
            
            # Set scheduler
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            print("✅ Pipeline and scheduler configured.")
            
            # Load IP-Adapter
            print("3/4 Loading IP-Adapter...")
            pipe.load_ip_adapter(
                self.ip_adapter_model_id, 
                subfolder="models", 
                weight_name="ip-adapter_sd15.bin"
            )
            print("✅ IP-Adapter loaded and attached to pipeline.")
            
            # Load Canny detector
            print("4/4 Loading Canny edge detector...")
            canny_detector = CannyDetector()
            print("✅ Canny detector loaded.")
            
            # Save models
            print("\n--- Saving Models for Reuse ---")
            models_data = {
                'pipe': pipe,
                'canny_detector': canny_detector,
                'device': self.device
            }
            
            models_path = os.path.join(self.models_dir, 'style_transfer_models.pkl')
            print(f"Saving models to: {models_path}")
            with open(models_path, 'wb') as f:
                pickle.dump(models_data, f)
            
            print(f"✅ Models saved successfully!")
            print("✅ All models loaded and saved successfully!")
            
            return models_data
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("This could be due to:")
            print("- Insufficient disk space (~3-5 GB needed)")
            print("- Network connectivity issues")
            print("- Missing dependencies")
            print("\nTry running again or check your internet connection.")
            return None
    
    def check_models_exist(self):
        """Check if models are already loaded and saved"""
        models_path = os.path.join(self.models_dir, 'style_transfer_models.pkl')
        return os.path.exists(models_path)
    
    def load_saved_models(self):
        """Load previously saved models"""
        models_path = os.path.join(self.models_dir, 'style_transfer_models.pkl')
        
        if not os.path.exists(models_path):
            raise FileNotFoundError("No saved models found. Please run load_all_models() first.")
        
        print("Loading saved models...")
        with open(models_path, 'rb') as f:
            models_data = pickle.load(f)
        
        print("✅ Saved models loaded successfully!")
        return models_data

def main():
    """Main function to load and save all models"""
    print("🚀 Starting AIMS Style Transfer Model Setup")
    print("=" * 50)
    
    loader = ModelLoader()
    
    # Check if models are already loaded
    if loader.check_models_exist():
        print("📁 Models already downloaded and saved.")
        print("📥 Loading models from saved files...")
        models = loader.load_saved_models()
        if models:
            print("✅ Models loaded from saved files successfully!")
        else:
            print("❌ Failed to load saved models. Re-downloading...")
            models = loader.load_all_models()
    else:
        print("📦 Models not found. Downloading and setting up...")
        print("⏳ This will download ~3-5 GB of data. Please be patient...")
        models = loader.load_all_models()
    
    if models:
        print("\n" + "=" * 50)
        print("🎉 SUCCESS: Models ready for use!")
        print(f"🖥️  Device: {models['device']}")
        print(f"📁 Models saved in: {loader.models_dir}")
        print("🚀 You can now run amalgam.py to start the FastAPI service.")
        print("💡 Command: python amalgam.py")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ FAILED: Could not load models.")
        print("🔍 Please check the error messages above.")
        print("💡 Common solutions:")
        print("   - Check internet connection")
        print("   - Ensure ~5GB free disk space")
        print("   - Try running again")
        print("=" * 50)

if __name__ == "__main__":
    main()
