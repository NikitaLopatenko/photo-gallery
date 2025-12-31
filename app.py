import streamlit as st
import os
import pickle
from sentence_transformers import SentenceTransformer, util

# 1. Page Config
st.set_page_config(page_title="Smart Family Gallery", page_icon="🧠", layout="wide")
st.title("🧠 AI Photo Search")
st.write("Type anything: 'dog', 'birthday', 'red car', 'happy family'")

# 2. Load Resources (Cached so it doesn't reload every time)
@st.cache_resource
def load_resources():
    # Load the model only for TEXT encoding (lighter)
    model = SentenceTransformer('clip-ViT-B-32')
    
    # Load the pre-computed image vectors
    if os.path.exists('features.pkl'):
        with open('features.pkl', 'rb') as f:
            image_data = pickle.load(f)
        return model, image_data
    return model, {}

try:
    model, image_data = load_resources()
except Exception as e:
    st.error("Could not load AI model. Make sure features.pkl is uploaded!")
    st.stop()

# 3. Search Logic
IMAGE_FOLDER = 'images'
query = st.text_input("🔍 Search:", "")

# Filter images
filtered_images = []

if not query:
    # If empty, show all images
    filtered_images = [(img, 1.0) for img in image_data.keys()]  # 100% match for all when no query
else:
    # THE AI MAGIC:
    # 1. Convert user text to vector
    text_embedding = model.encode(query)
    
    # 2. Compare text vector to all image vectors
    results = []
    for filename, img_embedding in image_data.items():
        # Calculate similarity score (higher is better)
        score = util.cos_sim(text_embedding, img_embedding).item()
        results.append((filename, score))
    
    # 3. Sort by score and keep best matches (e.g., score > 0.2)
    results.sort(key=lambda x: x[1], reverse=True)
    filtered_images = [(img, score) for img, score in results if score > 0.20]

# 4. Display
if not filtered_images:
    st.warning("No matching photos found.")
else:
    st.caption(f"Found {len(filtered_images)} matches")
    cols = st.columns(3)
    for idx, (filename, score) in enumerate(filtered_images):
        path = os.path.join(IMAGE_FOLDER, filename)
        if os.path.exists(path):
            with cols[idx % 3]:
                st.image(path, use_container_width=True)
                match_percentage = int(score * 100)
                st.caption(f"Match: {match_percentage}%")