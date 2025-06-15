# app.py
import streamlit as st
from PIL import Image, ImageFilter
import torch
from transformers import CLIPProcessor, CLIPModel
from model import FlexibleCLIPClassifier

# Load CLIP model + processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# UI layout
st.set_page_config(page_title="AI Content Moderation", layout="centered")
st.title("🛡️ AI Harm-Level Classifier")
st.markdown("Upload an image and/or type a caption. The AI will predict the harm level and suggest moderation.")

uploaded_file = st.file_uploader("Upload meme image (optional)", type=["png", "jpg", "jpeg"])
caption = st.text_area("Enter meme caption (optional)", "")

# Determine input mode
if uploaded_file and caption:
    input_mode = "both"
elif uploaded_file:
    input_mode = "image"
elif caption:
    input_mode = "text"
else:
    input_mode = None

# Load model based on mode
if input_mode:
    model = FlexibleCLIPClassifier(clip_model, mode=input_mode).to(device)
    model.load_state_dict(torch.load("full_new_model.pt", map_location=device))
    model.eval()

def moderate_content(level):
    return [
        "Low Harm: Allow or log",
        "Medium Harm: Blur/sanitize and notify",
        "High Harm: Alert + Auto-delete after grace period"
    ][level]

if input_mode:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = None

    # Prepare input
    if input_mode == "both":
        inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
    elif input_mode == "image":
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    elif input_mode == "text":
        inputs = clip_processor(text=[caption], return_tensors="pt", padding=True)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs, dim=1).item()

    harm_label = ["Low", "Medium", "High"][pred]
    action = moderate_content(pred)

    # Output
    st.markdown(f"Predicted Harm Level: **{harm_label}**")
    st.markdown(f"Moderation Action: {action}")

    if image:
        if pred >= 1:
            st.image(image.filter(ImageFilter.GaussianBlur(radius=12)), caption="Blurred for moderation")
        else:
            st.image(image, caption="Safe content")

else:
    st.info("Upload an image or enter a caption to begin.")
