import streamlit as st
import requests
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import pandas as pd
from pathlib import Path
import time

# Configure page
st.set_page_config(
    page_title="Histopathology Analysis",
    page_icon="🔬",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

class HistopathologyApp:
    def __init__(self):
        self.API_URL = "http://localhost:8000"
        
    def main(self):
        st.title("🔬 Histopathology Image Analysis")
        
        # Sidebar
        self.setup_sidebar()
        
        # Main content
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            self.image_upload_section()
            
        with col2:
            self.results_section()
    
    def setup_sidebar(self):
        st.sidebar.title("Settings")
        
        # Analysis settings
        st.sidebar.subheader("Analysis Settings")
        self.confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Visualization settings
        st.sidebar.subheader("Visualization Settings")
        self.show_attention = st.sidebar.checkbox(
            "Show Attention Heatmap",
            value=True
        )
        
        # Model info
        st.sidebar.subheader("Model Information")
        if st.sidebar.button("Load Model Info"):
            self.display_model_info()
    
    def image_upload_section(self):
        st.subheader("Upload Images")
        
        uploaded_file = st.file_uploader(
            "Choose a histopathology image",
            type=["png", "jpg", "jpeg", "tif", "tiff"]
        )
        
        if uploaded_file is not None:
            try:
                # Display image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Process button
                if st.button("Analyze Image"):
                    self.process_image(uploaded_file, image)
                    
            except Exception as e:
                st.error(f"Error opening image: {str(e)}")
    
    def process_image(self, file, image):
        with st.spinner("Analyzing image..."):
            try:
                # Save image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                # Prepare file for upload
                files = {
                    "file": (
                        file.name,
                        img_byte_arr.getvalue(),
                        "image/png"
                    )
                }
                
                # Make API request
                response = requests.post(
                    f"{self.API_URL}/predict",
                    files=files
                )
                response.raise_for_status()
                
                # Process response
                results = response.json()
                st.session_state.current_results = results
                
                if results.get('status') == 'success':
                    st.success("Analysis complete!")
                else:
                    st.warning(results.get('message', 'Analysis completed with warnings'))
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    def results_section(self):
        if 'current_results' not in st.session_state:
            st.info("Upload and analyze an image to see results")
            return
        
        results = st.session_state.current_results
        
        # Display predictions
        st.subheader("Analysis Results")
        
        if 'predictions' in results:
            # Confidence score
            confidence = results['predictions']['confidence']
            predicted_class = results['predictions']['predicted_class']
            class_name = "Malignant" if predicted_class == 1 else "Benign"
            
            st.metric(
                "Prediction",
                class_name,
                f"Confidence: {confidence:.1%}"
            )
            
            # Class probabilities
            probs = results['predictions']['class_probabilities']
            fig = go.Figure(data=[
                go.Bar(
                    x=['Benign', 'Malignant'],
                    y=probs,
                    marker_color=['#2ecc71', '#e74c3c']
                )
            ])
            fig.update_layout(
                title="Class Probabilities",
                yaxis_title="Probability",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Attention heatmap
            if self.show_attention and 'visualizations' in results:
                st.subheader("Attention Heatmap")
                if 'attention_heatmap' in results['visualizations']:
                    heatmap = np.array(results['visualizations']['attention_heatmap'])
                    st.image(heatmap, caption="Regions of Interest")
        
        # Display metadata
        if 'metadata' in results:
            st.subheader("Image Information")
            st.json(results['metadata'])
    
    def display_model_info(self):
        try:
            response = requests.get(f"{self.API_URL}/model-info")
            response.raise_for_status()
            
            info = response.json()
            st.sidebar.json(info)
            
        except Exception as e:
            st.sidebar.error(f"Error fetching model info: {str(e)}")

if __name__ == "__main__":
    app = HistopathologyApp()
    app.main()