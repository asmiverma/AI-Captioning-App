# AI-Powered Image Captioning Tool

A professional, interactive web application that uses a state-of-the-art Vision Transformer (Salesforce BLIP) to generate accurate, human-like captions for any uploaded image.

This project demonstrates a modern, end-to-end AI workflow, from implementing a large-scale pre-trained model to deploying it in a user-friendly Streamlit interface.

### Features

Upload Any Image: Supports .jpg, .jpeg, and .png file types.

AI-Powered Captions: Generates accurate, relevant, and natural-sounding captions.

Simple Web UI: Built with Streamlit for a clean, fast, and responsive user experience.

State-of-the-Art Model: Powered by the Salesforce BLIP model, a powerful Vision Transformer (ViT) from the Hugging Face hub.

### Tech Stack

**Python**: The core programming language.

**Streamlit**: For building and running the interactive web application.

**Hugging Face transformers**: To download and use the pre-trained BLIP model.

**PyTorch**: The deep learning framework that the model runs on.

**Pillow (PIL)**: For image processing and handling.

### How to Run This Project Locally

Follow these steps to get the application running on your own machine.

#### 1. Clone the Repository

```
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name 
``` 


#### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage your project's dependencies.

### Create the environment
```
python -m venv venv
```

### Activate the environment
```
1. On Windows:
.\venv\Scripts\activate
2. On macOS/Linux:
source vVenv/bin/activate
```

#### 3. Install the Required Libraries

All required libraries are listed in the requirements.txt file.
```
pip install -r requirements.txt
```

#### 4. Run the Streamlit App

Once the libraries are installed, you can run the application with a single command:
```
streamlit run app.py
```

The application will automatically open in your default web browser!

### How It Works

This project leverages transfer learning by using a large, pre-trained model.

**Model Loading**: The Salesforce/blip-image-captioning-large model is downloaded from Hugging Face. Streamlit's @st.cache_resource decorator cleverly loads this large model into memory only once, keeping the app fast.

**Image Processing**: When you upload an image, the BlipProcessor prepares it. It resizes, normalizes, and converts the image into a numerical format (tensors) that the AI model can understand.

**Caption Generation**: The BlipForConditionalGeneration model (a vision-encoder-decoder) analyzes the image's visual features and generates a sequence of text tokens as the caption.

**Decoding**: This sequence of tokens is decoded back into a human-readable string and displayed to the user.