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
    T.Resize(256), # Making images a uniform size (model wants a consistent size)
    T.CenterCrop(224), # centers the image making it to 224x224
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
        images.append(transform(img)) # adds all the images to a new list when they are transformed
        labels.append(fname.split(".")[0]) # gets the file name minus the .jpg

images = torch.stack(images).to(device)

# --- 4. Extract embeddings ---
with torch.no_grad():
    embeddings = model(images) # actually get the vector embedding from the images

embeddings = embeddings.cpu().numpy()

# --- 5. Dimensionality reduction (TSNE for visualization) --- (This is like feature projection/dimensionality reduction in QEA 1)
# OMG, I remember orthogonal projection, getting as close as you can to the value (which you can't get to), so you project it to a dimension you can
# Low perplexity only looks at a couple neighbors (other vectors with similar values)
# High perplexity looks at a lot of neighbors
# Low perplexity will group a lot closer together (not getting the global picture)
# too high of perplexity will just put all the vectors together in a circle because it considered too many vectors and got closer to them.

# Low perplexity: Creates many tight, small clusters (hyper-local focus)
# Medium perplexity: Creates meaningful clusters with good separation
# High perplexity: Mushes everything together (loses all structure)
tsne = TSNE(n_components=2, random_state=42, perplexity=2) # perplexity is just how many neighbors it considers (which obviously means don't put it 
#higher than the amount of images that there actually are)
embeddings_2d = tsne.fit_transform(embeddings)

# --- 6. Plot ---
plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    x, y = embeddings_2d[i]
    plt.scatter(x, y, label=label)
    plt.text(x+0.5, y+0.5, label, fontsize=9)

plt.title("DINOv2 Embedding Clusters (t-SNE)")
plt.show()
