import streamlit as st
import os

# 1. Page Configuration (Tab Title & Icon)
st.set_page_config(page_title="My Photo Gallery", page_icon="📸", layout="wide")

# 2. Title and Description
st.title("📸 My Photo Gallery")
st.write("A collection of my favorite moments.")

# 3. Path to your images
IMAGE_FOLDER = 'images'

# 4. Logic to find images
def load_images():
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER) # Creates the folder if it doesn't exist
        return []
    
    # Get all files that end with .png, .jpg, or .jpeg
    files = os.listdir(IMAGE_FOLDER)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return image_files

images = load_images()

# 5. Display Images
if not images:
    st.info("No photos found yet! Upload them to the 'images' folder.")
else:
    # Create a grid of columns (adjust the number 3 for more/less columns)
    cols = st.columns(3) 
    for idx, image_file in enumerate(images):
        with cols[idx % 3]: # Cycles through columns 0, 1, 2
            st.image(os.path.join(IMAGE_FOLDER, image_file), use_container_width=True)
            st.caption(f"Photo {idx + 1}")