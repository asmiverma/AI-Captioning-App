import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# --- CONFIGURATION ---
# Set the device to use. Use 'cuda' if you have a compatible GPU for faster processing.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/blip-image-captioning-large"

# --- MODEL LOADING ---
@st.cache_resource # This decorator caches the model so it doesn't reload on every interaction.
def load_model():
    """Loads the BLIP image captioning model and processor from Hugging Face."""
    print("Loading AI model... This may take a few minutes the first time.")
    # The processor prepares the image for the model.
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    # The model itself is a powerful Vision Transformer.
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Model loaded successfully.")
    return processor, model

# --- MAIN APPLICATION ---
def main():
    # --- PAGE SETUP ---
    # This part now runs immediately, so you won't see a black screen.
    st.set_page_config(page_title="AI Image Captioning Tool", layout="wide")
    st.title("🖼️ AI-Powered Image Captioning Tool")
    st.markdown("Upload an image, and the AI will describe it for you in plain English. This uses a state-of-the-art Vision Transformer (ViT) model called Salesforce BLIP.")
    st.info("The AI model is loading in the background. This may take a moment on the first run...")

    # --- LOAD THE AI MODEL ---
    # We moved this down. The app will show the title first, then load the model.
    processor, model = load_model()
    st.success("AI Model Loaded Successfully!") # This message will appear when ready.


    # --- IMAGE UPLOADER ---
    st.header("Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # If a file is uploaded, we display it.
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your Uploaded Image", use_column_width=True)
        
        # Add a button to trigger the caption generation.
        if st.button("Generate Caption", key="generate"):
            with st.spinner("🧠 The AI is thinking..."):
                # --- CAPTION GENERATION LOGIC ---
                # 1. Prepare the image for the model using the processor.
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                
                # 2. Generate the caption using the model.
                # The model will output a sequence of token IDs.
                outputs = model.generate(**inputs, max_length=50)
                
                # 3. Decode the token IDs back into a human-readable string.
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                
                # --- DISPLAY THE RESULT ---
                st.subheader("🤖 AI-Generated Caption:")
                st.write(f"### {caption.capitalize()}")

if __name__ == "__main__":
    main()