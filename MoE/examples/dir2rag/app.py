import streamlit as st
import pandas as pd
import json
from chromadb import PersistentClient
import os
from dev_tools.enums.embeddings import Embeddings
from langchain_core.documents import Document


# Set page config
st.set_page_config(
    page_title="ArXiv PE",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS for cards
st.markdown("""
<style>
    .paper-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .paper-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    p {
        color: #333;
    }
    h3 {
        color: #333;
    }
        
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chroma_data():
    # Get Chroma collection
    client = PersistentClient(path="/Users/dabbos/mlv0rr/research/retrieval/arxiv/daily/chroma")
    collection = client.get_collection("abstracts_test1")
    ids = collection.get()['ids']
    docs = []
    
    for id in ids:
        doc = collection.get([id])
        docs.append(doc)
    
    return docs

def display_papers(docs):
    # Initialize selected papers list
    if 'selected_papers' not in st.session_state:
        st.session_state.selected_papers = []

    for idx, doc in enumerate(docs):
        content = doc['documents'][0]
        metadata = doc['metadatas'][0]
        col1, col2 = st.columns([0.1, 0.9])

        with col1:
            selected = st.checkbox("", key=f"checkbox_{idx}")
            if selected and idx not in st.session_state.selected_papers:
                st.session_state.selected_papers.append(idx)
            elif not selected and idx in st.session_state.selected_papers:
                st.session_state.selected_papers.remove(idx)
        
        with col2:
            with st.container():
                st.markdown(f"""
                <div class="paper-card">
                    <h3 style="color: #333;">{metadata['title']}</h3>
                    <p style="color: #333;"><strong>Paper ID:</strong> {metadata['id']}</p>
                    <p style="color: #333;"><strong>PDF:</strong> <a href="{metadata['pdf_url']}" target="_blank">Open Paper on Arxiv</a></p>
                    <p style="color: #333;"><strong>Highlights:</strong></p>
                    <p style="color: #333; white-space: normal">{metadata.get('highlights', '').replace('**', '').replace('\n\n', '\n').replace('*', '\n•')}</p>
                    <details>
                        <summary style="color: #333;">Abstract</summary>
                        <p style="color: #333;">{content}</p>
                    </details>
                </div>
                """, unsafe_allow_html=True)

    # Display selected papers summary
    if st.session_state.selected_papers:
        st.sidebar.header("Selected Papers")
        for idx in st.session_state.selected_papers:
            metadata = docs[idx]['metadatas'][0]
            st.sidebar.markdown(f"""
            * **{metadata['title']}**
            * ID: {metadata['id']}
            * PDF: {metadata['pdf_url']}
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    docs = load_chroma_data()
    display_papers(docs)

  