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
import base64
from typing import Dict, List

# Configure page
st.set_page_config(
    page_title="Histopathology Analysis",
    page_icon="🔬",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
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
        self.show_graph = st.sidebar.checkbox(
            "Show Tissue Graph",
            value=True
        )
        
        # Model info
        st.sidebar.subheader("Model Information")
        if st.sidebar.button("Load Model Info"):
            self.display_model_info()
    
    def image_upload_section(self):
        st.subheader("Upload Images")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose histopathology image(s)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                image = Image.open(file)
                st.image(
                    image,
                    caption=f"Uploaded: {file.name}",
                    use_column_width=True
                )
                
                if st.button(f"Analyze {file.name}"):
                    self.process_image(file, image)
    
    def results_section(self):
        if 'current_results' not in st.session_state:
            st.info("Upload and analyze an image to see results")
            return
        
        results = st.session_state.current_results
        
        # Display predictions
        st.subheader("Analysis Results")
        
        # Confidence score
        confidence = results['predictions']['confidence']
        predicted_class = results['predictions']['predicted_class']
        
        st.metric(
            "Prediction Confidence",
            f"{confidence:.2%}",
            delta="Above threshold" if confidence > self.confidence_threshold else "Below threshold"
        )
        
        # Class probabilities
        fig_probs = go.Figure(data=[
            go.Bar(
                x=['Benign', 'Malignant'],
                y=results['predictions']['class_probabilities'],
                marker_color=['#2ecc71', '#e74c3c']
            )
        ])
        fig_probs.update_layout(
            title="Class Probabilities",
            yaxis_title="Probability",
            showlegend=False
        )
        st.plotly_chart(fig_probs, use_container_width=True)
        
        # Attention heatmap
        if self.show_attention:
            st.subheader("Attention Heatmap")
            st.image(
                results['visualizations']['attention_heatmap'],
                caption="Regions of Interest",
                use_column_width=True
            )
        
        # Download results
        if st.button("Download Results"):
            self.download_results(results)
    
    def process_image(self, file, image):
        with st.spinner("Analyzing image..."):
            try:
                response = requests.post(
                    f"{self.API_URL}/predict",
                    files={"file": file}
                )
                response.raise_for_status()
                
                results = response.json()
                st.session_state.current_results = results
                st.success("Analysis complete!")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    def display_model_info(self):
        try:
            response = requests.get(f"{self.API_URL}/model-info")
            response.raise_for_status()
            
            info = response.json()
            st.sidebar.json(info)
            
        except Exception as e:
            st.sidebar.error(f"Error fetching model info: {str(e)}")
    
    def download_results(self, results: Dict):
        # Convert results to DataFrame
        df = pd.DataFrame({
            'Metric': ['Confidence', 'Predicted Class'],
            'Value': [
                results['predictions']['confidence'],
                results['predictions']['predicted_class']
            ]
        })
        
        # Create Excel buffer
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Add visualizations
            worksheet = writer.sheets['Results']
            
            # Add attention heatmap
            img_data = results['visualizations']['attention_heatmap']
            worksheet.insert_image('D2', 'heatmap.png', {'image_data': img_data})
        
        # Offer download
        buffer.seek(0)
        st.download_button(
            label="Download Results (Excel)",
            data=buffer,
            file_name="histopathology_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    app = HistopathologyApp()
    app.main()