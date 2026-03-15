# AI-Powered Image Captioning Web App

A clean, deployable AI project that generates natural language captions from uploaded images using Salesforce BLIP.

This repository is designed to be recruiter-friendly: practical AI integration, clear architecture, and readable code.

## Project Overview

This app demonstrates a complete inference workflow:

1. User uploads an image in the Streamlit UI.
2. Image is processed with Pillow.
3. BLIP processor converts image into tensors.
4. BLIP model generates token IDs for a caption.
5. Tokens are decoded into readable text.
6. Caption is shown and can be downloaded.

## Demo Screenshot

Add your screenshot at `assets/demo_screenshot.png`, then uncomment the line below:

<!-- ![App Demo](assets/demo_screenshot.png) -->

## Tech Stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Salesforce BLIP (`Salesforce/blip-image-captioning-base`)
- Pillow

## How It Works

### High-Level Architecture (for Recruiters)

The project uses a simple modular architecture:

- `app.py`: Streamlit presentation layer and user interactions.
- `model.py`: AI model loading and caption generation functions.
- `utils.py`: Image/file helper utilities.

This separation keeps the UI code clean while making model logic easier to test and maintain.

### Inference Flow

1. `app.py` loads BLIP once using `@st.cache_resource`.
2. User uploads an image (or chooses one from `example_images/`).
3. `utils.py` converts the image to RGB PIL format.
4. `model.py` runs BLIP inference on CPU/GPU.
5. The generated caption is displayed and offered as a `.txt` download.

## Project Structure

```text
AI-Captioning-App/
├── app.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
├── assets/
│   └── README.md
└── example_images/
	└── README.md
```

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/asmiverma/AI-Captioning-App.git
cd AI-Captioning-App
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the app:

```bash
streamlit run app.py
```

## Future Improvements

- Support beam search/temperature options in the UI for caption diversity.
- Add multilingual caption translation.
- Add lightweight evaluation metrics over a small benchmark dataset.
- Package with Docker for one-command deployment.
