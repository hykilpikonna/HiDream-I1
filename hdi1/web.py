import torch
import gradio as gr
import logging
import os
import tempfile
import glob
from datetime import datetime
from PIL import Image
from .nf4 import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory for saving images
OUTPUT_DIR = os.path.join("outputs")

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Scheduler options (flow-matching only)
SCHEDULER_OPTIONS = [
    "FlashFlowMatchEulerDiscreteScheduler",
    "FlowUniPCMultistepScheduler"
]

# Image format options
IMAGE_FORMAT_OPTIONS = ["PNG", "JPEG", "WEBP"]

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    try:
        return tuple(map(int, resolution_str.split("(")[0].strip().split(" × ")))
    except (ValueError, IndexError) as e:
        raise ValueError("Invalid resolution format") from e

def clean_previous_temp_files():
    """Delete temporary files from previous generations matching hdi1_* pattern and log Gradio temp files."""
    temp_dir = tempfile.gettempdir()
    patterns = [os.path.join(temp_dir, f"hdi1_*.{ext}") for ext in ["png", "jpeg", "webp"]]
    deleted_files = []
    
    # Clean hdi1_* files
    for pattern in patterns:
        for temp_file in glob.glob(pattern):
            try:
                os.remove(temp_file)
                deleted_files.append(temp_file)
                logger.info(f"Deleted temporary file: {temp_file}")
            except OSError as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")
    
    # Log Gradio temp files (for monitoring)
    gradio_temp_dir = os.path.join(temp_dir, "gradio")
    if os.path.exists(gradio_temp_dir):
        for root, _, files in os.walk(gradio_temp_dir):
            for file in files:
                if file.endswith((".png", ".jpeg", ".webp")):
                    gradio_file = os.path.join(root, file)
                    logger.info(f"Found Gradio temporary file: {gradio_file}")
    
    return deleted_files

def clean_all_temp_files():
    """Manually clean hdi1_* and Gradio temporary files, with user confirmation."""
    status_message = "Starting temporary file cleanup..."
    logger.info(status_message)
    
    try:
        # Clean hdi1_* files
        deleted_files = clean_previous_temp_files()
        
        # Clean Gradio temp files
        temp_dir = tempfile.gettempdir()
        gradio_temp_dir = os.path.join(temp_dir, "gradio")
        if os.path.exists(gradio_temp_dir):
            for root, _, files in os.walk(gradio_temp_dir):
                for file in files:
                    if file.endswith((".png", ".jpeg", ".webp")):
                        gradio_file = os.path.join(root, file)
                        try:
                            os.remove(gradio_file)
                            deleted_files.append(gradio_file)
                            logger.info(f"Deleted Gradio temporary file: {gradio_file}")
                        except OSError as e:
                            logger.warning(f"Failed to delete Gradio temporary file {gradio_file}: {str(e)}")
        
        status_message = f"Cleanup complete. Deleted {len(deleted_files)} files."
        logger.info(status_message)
        return status_message
    except Exception as e:
        error_message = f"Cleanup error: {str(e)}"
        logger.error(error_message)
        return error_message

def gen_img_helper(model, prompt, res, seed, scheduler, guidance_scale, num_inference_steps, shift, image_format):
    global pipe, current_model
    status_message = "Starting image generation..."

    try:
        # Clean up previous temporary files
        status_message = "Cleaning up previous temporary files..."
        logger.info(status_message)
        clean_previous_temp_files()
        status_message = "Previous temporary files cleaned."

        # Validate inputs
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        if not isinstance(seed, (int, float)) or seed < -1:
            raise ValueError("Seed must be -1 or a non-negative integer")
        if num_inference_steps < 1 or num_inference_steps > 100:
            raise ValueError("Number of inference steps must be between 1 and 100")
        if guidance_scale < 0 or guidance_scale > 10:
            raise ValueError("Guidance scale must be between 0 and 10")
        if shift < 1 or shift > 10:
            raise ValueError("Shift must be between 1 and 10")

        # 1. Check if the model matches loaded model, load the model if not
        if model != current_model:
            status_message = f"Unloading model {current_model}..."
            logger.info(status_message)
            if pipe is not None:
                del pipe
                torch.cuda.empty_cache()
            
            status_message = f"Loading model {model}..."
            logger.info(status_message)
            pipe, _ = load_models(model)
            current_model = model
            status_message = "Model loaded successfully!"
            logger.info(status_message)

        # 2. Update scheduler
        config = MODEL_CONFIGS[model]
        scheduler_map = {
            "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler,
            "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler
        }
        if scheduler not in scheduler_map:
            raise ValueError(f"Invalid scheduler: {scheduler}")
        scheduler_class = scheduler_map[scheduler]
        device = pipe._execution_device

        # Set scheduler with shift for flow-matching schedulers
        pipe.scheduler = scheduler_class(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)

        # 3. Generate image
        status_message = "Generating image..."
        logger.info(status_message)
        res = parse_resolution(res)
        image, seed = generate_image(pipe, model, prompt, res, seed, guidance_scale, num_inference_steps)
        
        # 4. Save image locally with selected format
        status_message = "Saving image locally..."
        logger.info(status_message)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_extension = image_format.lower()
        output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.{file_extension}")
        if image_format == "JPEG":
            image = image.convert("RGB")  # JPEG doesn't support RGBA
        image.save(output_path, format=image_format)
        logger.info(f"Image saved to {output_path}")
        
        # 5. Prepare image for download in selected format
        status_message = "Preparing image for download..."
        logger.info(status_message)
        download_filename = f"generated_image_{timestamp}.{file_extension}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}", prefix="hdi1_") as temp_file:
            if image_format == "JPEG":
                image = image.convert("RGB")  # Ensure JPEG compatibility
            image.save(temp_file, format=image_format)
            temp_file_path = temp_file.name
        logger.info(f"Temporary file created at {temp_file_path}")
        
        status_message = "Image generation complete!"
        logger.info(status_message)
        return image, seed, f"Image saved to: {output_path}", temp_file_path, status_message

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(error_message)
        return None, None, None, None, error_message

def generate_image(pipe, model_type, prompt, resolution, seed, guidance_scale, num_inference_steps):
    try:
        # Parse resolution
        width, height = resolution

        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Common parameters
        params = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": 1,
            "generator": generator
        }

        images = pipe(**params).images
        return images[0], seed
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}") from e

if __name__ == "__main__":
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    # Initialize globals without loading model
    current_model = None
    pipe = None

    # Create Gradio interface
    with gr.Blocks(title="HiDream-I1-nf4 Dashboard") as demo:
        gr.Markdown("# HiDream-I1-nf4 Dashboard")
        gr.Markdown("**Note**: Use the 'Download Image' link below to download the image in your selected format (PNG, JPEG, or WEBP). Downloading from the image preview's download button is WEBP format.")
        
        with gr.Row():
            with gr.Column():
                model_type = gr.Radio(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="fast",
                    label="Model Type",
                    info="Select model variant (e.g., 'fast' for quick generation)"
                )
                
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="A cat holding a sign that says \"Hi-Dreams.ai\".", 
                    lines=3
                )
                
                resolution = gr.Radio(
                    choices=RESOLUTION_OPTIONS,
                    value=RESOLUTION_OPTIONS[0],
                    label="Resolution",
                    info="Select image resolution"
                )
                
                seed = gr.Number(
                    label="Seed (use -1 for random)", 
                    value=-1, 
                    precision=0
                )
                
                scheduler = gr.Radio(
                    choices=SCHEDULER_OPTIONS,
                    value="FlashFlowMatchEulerDiscreteScheduler",
                    label="Scheduler",
                    info="Select scheduler type. Flow-matching schedulers are optimized for HiDream, providing stable, high-quality, prompt-relevant images."
                )
                
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=2.0,
                    label="Guidance Scale",
                    info="Controls prompt adherence. Use 2.0–5.0; increase to 4.0–5.0 for stronger prompt following."
                )
                
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=25,
                    label="Number of Inference Steps",
                    info="Controls denoising steps. Use 25–50; increase to 40–50 for sharper images."
                )
                
                shift = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=3.0,
                    label="Shift",
                    info="Scheduler shift parameter for flow-matching schedulers. Use 1.0–5.0; 3.0 is a good default."
                )
                
                image_format = gr.Radio(
                    choices=IMAGE_FORMAT_OPTIONS,
                    value="PNG",
                    label="Image Format",
                    info="Select the format for the saved and downloaded image."
                )
                
                generate_btn = gr.Button("Generate Image")
                cleanup_btn = gr.Button("Clean Temporary Files")
                
            with gr.Column():
                status_message = gr.Textbox(label="Status", value="Ready", interactive=False)
                output_image = gr.Image(label="Generated Image", type="pil")
                seed_used = gr.Number(label="Seed Used", interactive=False)
                save_path = gr.Textbox(label="Saved Image Path", interactive=False)
                download_file = gr.File(label="Download Image", interactive=False, file_types=[".png", ".jpeg", ".webp"])
        
        generate_btn.click(
            fn=gen_img_helper,
            inputs=[model_type, prompt, resolution, seed, scheduler, guidance_scale, num_inference_steps, shift, image_format],
            outputs=[output_image, seed_used, save_path, download_file, status_message]
        )
        cleanup_btn.click(
            fn=clean_all_temp_files,
            inputs=[],
            outputs=[status_message]
        )

    demo.launch(share=True, pwa=True)
