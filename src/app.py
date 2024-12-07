import streamlit as st
from PIL import Image
import os
from search_backend import ImageSearchEngine
from model_utils import load_clip_model, encode_text, encode_image
import numpy as np

# Load model and data
model, preprocess, tokenizer = load_clip_model()
image_folder = "data/coco_images_resized"
engine = ImageSearchEngine("data/image_embeddings.pickle", image_folder, model)

st.title("Simplified Google Image Search")

text_query = st.text_input("Enter a text query:", "")
uploaded_image = st.file_uploader("Upload an image query:", type=["jpg","jpeg","png"])
text_weight = st.slider("Weight for text query (0.0 = only image, 1.0 = only text)", 0.0, 1.0, 0.5)

use_pca = st.checkbox("Use PCA embeddings?")
pca_k = 512
if use_pca:
    pca_k = st.number_input("Number of principal components (1 to 512):", min_value=1, max_value=512, value=100)

if st.button("Search"):
    # Compute query embedding
    text_emb = None
    img_emb = None

    if text_query:
        text_emb = encode_text(model, tokenizer, text_query)

    if uploaded_image:
        user_img = Image.open(uploaded_image)
        img_emb = encode_image(model, preprocess, user_img)

    if text_emb is not None and img_emb is not None:
        # Hybrid query
        query_embedding = text_weight * text_emb + (1 - text_weight) * img_emb
    elif text_emb is not None:
        query_embedding = text_emb
    elif img_emb is not None:
        query_embedding = img_emb
    else:
        st.warning("Please provide a text query and/or upload an image.")
        query_embedding = None

    if query_embedding is not None:
        # Normalize combined query embedding
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        results = engine.search(query_embedding, top_k=5, use_pca=use_pca, pca_k=pca_k)
        for idx, row in results.iterrows():
            fname = row['file_name']
            sim = row['similarity']
            img_path = os.path.join(image_folder, fname)
            st.image(img_path, caption=f"Similarity: {sim:.4f}")
