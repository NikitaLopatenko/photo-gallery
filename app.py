import streamlit as st
import os
import pickle
import json
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# 1. Page Config
st.set_page_config(page_title="Семейная Галерея", page_icon="📸", layout="wide")

# Constants
IMAGE_FOLDER = 'images'
TAGS_FILE = 'image_tags.json'
FEATURES_FILE = 'features.pkl'

# 2. Load Resources (Cached so it doesn't reload every time)
@st.cache_resource
def load_resources():
    # Load the model
    model = SentenceTransformer('clip-ViT-B-32')
    
    # Load the pre-computed image vectors
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, 'rb') as f:
            image_data = pickle.load(f)
        return model, image_data
    return model, {}

# Load tags from JSON file
def load_tags():
    if os.path.exists(TAGS_FILE):
        with open(TAGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Save tags to JSON file
def save_tags(tags):
    with open(TAGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)

# Get all unique tags from tagged images
def get_all_tags(tags_dict):
    all_tags = set()
    for image_tags in tags_dict.values():
        all_tags.update(image_tags)
    return sorted(list(all_tags))

# Find similar images and suggest tags based on them
def suggest_tags(current_image, image_data, tags_dict, top_n=5):
    if current_image not in image_data:
        return []
    
    current_embedding = image_data[current_image]
    similarities = []
    
    for filename, embedding in image_data.items():
        if filename == current_image:
            continue
        if filename in tags_dict and tags_dict[filename]:  # Only consider tagged images
            score = util.cos_sim(current_embedding, embedding).item()
            similarities.append((filename, score, tags_dict[filename]))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Collect tags from most similar images
    suggested_tags = Counter()
    for filename, score, tags in similarities[:top_n]:
        for tag in tags:
            suggested_tags[tag] += score  # Weight by similarity
    
    # Return top suggested tags
    return [tag for tag, count in suggested_tags.most_common(10)]

try:
    model, image_data = load_resources()
except Exception as e:
    st.error("Не удалось загрузить AI модель. Убедитесь, что features.pkl существует!")
    st.stop()

# Load tags
tags_dict = load_tags()

# Sidebar navigation
st.sidebar.title("📸 Навигация")
page = st.sidebar.radio("Выберите режим:", ["🏷️ Тегирование", "🔍 Поиск"])

if page == "🏷️ Тегирование":
    st.title("🏷️ Тегирование Фотографий")
    st.write("Добавьте теги на русском языке к фотографиям. AI будет предлагать теги на основе похожих фотографий.")
    
    # Get all images
    all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_images.sort()
    
    if not all_images:
        st.warning("Нет фотографий в папке images/")
    else:
        # Image selector
        selected_image = st.selectbox("Выберите фотографию для тегирования:", all_images)
        
        if selected_image:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display image
                image_path = os.path.join(IMAGE_FOLDER, selected_image)
                if os.path.exists(image_path):
                    st.image(image_path, use_container_width=True)
            
            with col2:
                st.subheader("Теги для этой фотографии")
                
                # Get current tags
                current_tags = set(tags_dict.get(selected_image, []))
                
                # Get AI suggestions
                suggested_tags = suggest_tags(selected_image, image_data, tags_dict)
                
                # Show suggested tags as checkboxes
                if suggested_tags:
                    st.write("**AI предлагает следующие теги:**")
                    selected_suggestions = []
                    for tag in suggested_tags:
                        if st.checkbox(tag, value=tag in current_tags, key=f"suggest_{selected_image}_{tag}"):
                            selected_suggestions.append(tag)
                else:
                    st.info("Пока нет предложений. Начните тегировать другие фотографии, и AI будет учиться!")
                    selected_suggestions = []
                
                # Custom tag input
                st.write("**Или введите свой тег (на русском):**")
                custom_tags_input = st.text_input(
                    "Дополнительные теги (через запятую):",
                    value=", ".join(current_tags - set(suggested_tags)),
                    key=f"custom_{selected_image}"
                )
                
                # Parse custom tags
                custom_tags = [tag.strip() for tag in custom_tags_input.split(",") if tag.strip()]
                
                # Combine all tags
                all_selected_tags = list(set(selected_suggestions + custom_tags))
                
                # Show current tags
                if all_selected_tags:
                    st.write("**Текущие теги:**")
                    for tag in all_selected_tags:
                        st.write(f"• {tag}")
                
                # Save button
                if st.button("💾 Сохранить теги", type="primary"):
                    tags_dict[selected_image] = all_selected_tags
                    save_tags(tags_dict)
                    st.success(f"✅ Теги сохранены для {selected_image}!")
                    st.rerun()
                
                # Show statistics
                st.divider()
                total_tagged = len([img for img in tags_dict if tags_dict[img]])
                st.caption(f"Всего помечено фотографий: {total_tagged} / {len(all_images)}")

else:  # Search page
    st.title("🔍 Поиск по Тегам")
    st.write("Ищите фотографии по тегам на русском языке")
    
    # Get all tags
    all_tags = get_all_tags(tags_dict)
    
    if not all_tags:
        st.warning("Пока нет тегов. Перейдите в режим 'Тегирование' чтобы добавить теги к фотографиям.")
    else:
        # Search interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input("🔍 Поиск по тегам (введите теги через запятую):", "")
        
        with col2:
            st.write("**Доступные теги:**")
            if st.checkbox("Показать все теги"):
                for tag in all_tags:
                    st.write(f"• {tag}")
        
        # Filter images by tags
        filtered_images = []
        
        if not search_query:
            # Show all tagged images
            filtered_images = [(img, tags_dict[img]) for img in tags_dict if tags_dict[img]]
        else:
            # Search by tags
            search_tags = [tag.strip().lower() for tag in search_query.split(",") if tag.strip()]
            for img, img_tags in tags_dict.items():
                img_tags_lower = [t.lower() for t in img_tags]
                # Check if any search tag matches
                if any(st in img_tags_lower for st in search_tags):
                    filtered_images.append((img, img_tags))
        
        # Display results
        if not filtered_images:
            st.info("Нет фотографий, соответствующих вашему запросу.")
        else:
            st.caption(f"Найдено {len(filtered_images)} фотографий")
            cols = st.columns(3)
            for idx, (filename, img_tags) in enumerate(filtered_images):
                path = os.path.join(IMAGE_FOLDER, filename)
                if os.path.exists(path):
                    with cols[idx % 3]:
                        st.image(path, use_container_width=True)
                        # Show tags
                        tags_display = ", ".join(img_tags)
                        st.caption(f"🏷️ {tags_display}")
