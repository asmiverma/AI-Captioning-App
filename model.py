import torch
from PIL.Image import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def get_device() -> str:
    """Return the best available compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_blip_components(model_name: str):
    """Load BLIP processor and model from Hugging Face."""
    device = get_device()

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    return processor, model, device


@torch.inference_mode()
def generate_caption(
    image: Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: str,
    max_new_tokens: int = 40,
) -> str:
    """Generate a natural language caption for a PIL image."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    output_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens)

    caption = processor.decode(output_tokens[0], skip_special_tokens=True).strip()
    return caption
