import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import cv2
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load DINO
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
model.eval()

# Load your image (1000x1000)
img_path = "Bug/Device_1DC64544_InnerCam_2025-06-15 08_02_00_B1.jpg"
img = Image.open(img_path).convert("RGB")
w, h = img.size

# Process image
inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    # last_hidden_state: (1, num_patches+1, dim)
    # attention matrix: may vary per implementation; we can use the attention weights in ViT
    # For HF DINO, you can extract attention from the transformer blocks
    # For simplicity, we will use mean CLS attention from the last layer

    # patch embeddings
    patch_emb = outputs.last_hidden_state[0, 1:, :]  # skip CLS token
    num_patches = patch_emb.shape[0]
    grid_size = int(np.sqrt(num_patches))

    # Approximate attention map: L2 norm of each patch embedding
    # Patches that differ from background tend to have higher norm
    attn_map = torch.norm(patch_emb, dim=1).cpu().numpy()
    attn_map = attn_map.reshape(grid_size, grid_size)

# Upsample attention map to image size
attn_map_img = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)

# Threshold attention map to get candidate bugs
mask = (attn_map_img > attn_map_img.mean() * 1.2).astype(np.uint8)  # tweak threshold

# Connected components
num_labels, components = cv2.connectedComponents(mask)
output_img = np.array(img).copy()
bboxes = []

for i in range(1, num_labels):
    ys, xs = np.where(components == i)
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    bboxes.append([x1, y1, x2, y2])
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

print(f"Detected {len(bboxes)} bugs")

plt.figure(figsize=(8, 8))
plt.imshow(output_img)
plt.title("Bugs detected via DINO attention map")
plt.axis("off")
plt.show()
