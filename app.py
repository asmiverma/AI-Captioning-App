import streamlit as st

from model import generate_caption, load_blip_components
from utils import caption_to_download_text, list_example_images, load_image_file

MODEL_NAME = "Salesforce/blip-image-captioning-base"


@st.cache_resource(show_spinner=False)
def get_cached_model_components(model_name: str):
    """Cache heavy model artifacts so they are loaded only once per session."""
    return load_blip_components(model_name)


def render_sidebar() -> None:
    st.sidebar.header("Model Settings")
    st.sidebar.write(f"Model: `{MODEL_NAME}`")
    st.sidebar.caption("The app uses BLIP from Hugging Face for image caption generation.")


def main() -> None:
    st.set_page_config(page_title="AI Captioning App", page_icon="🖼️", layout="wide")

    st.title("AI-Powered Image Captioning Web App")
    st.write(
        "Upload an image and generate a natural language caption using "
        "Salesforce BLIP (Vision Transformer + language decoder)."
    )

    render_sidebar()

    with st.spinner("Loading BLIP model (first run may take ~1-2 minutes)..."):
        processor, model, device = get_cached_model_components(MODEL_NAME)
    st.success(f"Model ready on `{device}`")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    image = None
    if uploaded_file is not None:
        image = load_image_file(uploaded_file)
        st.image(image, caption="Uploaded image preview", use_container_width=True)

    example_images = list_example_images("example_images")
    if example_images:
        st.caption("Or try an image from `example_images/`.")
        selected_example = st.selectbox("Choose a demo image", options=example_images)
        if selected_example and image is None:
            image = load_image_file(selected_example)
            st.image(image, caption=f"Example preview: {selected_example.name}", use_container_width=True)

    if st.button("Generate Caption", type="primary", disabled=image is None):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image=image, processor=processor, model=model, device=device)

        st.subheader("Generated Caption")
        st.info(caption)

        st.download_button(
            label="Download Caption",
            data=caption_to_download_text(caption),
            file_name="generated_caption.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()