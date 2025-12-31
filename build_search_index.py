import os
import pickle
from sentence_transformers import SentenceTransformer
from PIL import Image

# 1. Setup
IMAGE_FOLDER = 'images'
OUTPUT_FILE = 'features.pkl'

# Load the AI model (CLIP)
print("Loading AI model... (This might take a minute)")
model = SentenceTransformer('clip-ViT-B-32')

# 2. Find Images
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_files)} images. Analyzing...")

# 3. Analyze Images (Turn them into vectors)
image_data = {}
for idx, file in enumerate(image_files):
    path = os.path.join(IMAGE_FOLDER, file)
    try:
        # The AI reads the image here
        img = Image.open(path)
        embedding = model.encode(img)
        image_data[file] = embedding
        print(f"[{idx+1}/{len(image_files)}] Processed {file}")
    except Exception as e:
        print(f"Skipping {file}: {e}")

# 4. Save the Brain
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(image_data, f)

print(f"✅ Done! Saved AI data to {OUTPUT_FILE}")