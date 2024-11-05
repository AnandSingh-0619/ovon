import torch
import clip
from PIL import Image
import numpy as np
from torch.nn import CosineSimilarity
# Set the device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and preprocessing pipeline
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
image = preprocess(Image.open("Sim1_m.png")).unsqueeze(0).to(device)

# Tokenize the text prompt
text = clip.tokenize(["a table"]).to(device)

# Compute image and text features
with torch.no_grad():
    image_features = model.encode_image(image).cpu().numpy()
    text_features = model.encode_text(text).cpu().numpy()

# Calculate the cosine similarity between the image and the text feature
    logits_per_image, logits_per_text = model(image, text)
    print(logits_per_image)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs) 