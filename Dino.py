import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# --- 1. Load DINO model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = dinov2_vits14.to(device)  
model.eval()

# --- 2. Preprocessing for images ---
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

# --- 3. Load your images ---
image_dir = "images"
images, labels = [], []
for fname in os.listdir(image_dir):
    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        images.append(transform(img))
        labels.append(fname.split(".")[0]) 

images = torch.stack(images).to(device)

# --- 4. Extract embeddings ---
with torch.no_grad():
    embeddings = model(images)

embeddings = embeddings.cpu().numpy()

# --- 5. Dimensionality reduction (TSNE for visualization) ---
tsne = TSNE(n_components=2, random_state=42, perplexity=14)
embeddings_2d = tsne.fit_transform(embeddings)

# --- 6. Plot ---
plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    x, y = embeddings_2d[i]
    plt.scatter(x, y, label=label)
    plt.text(x+0.5, y+0.5, label, fontsize=9)

plt.title("DINOv2 Embedding Clusters (t-SNE)")
plt.show()
