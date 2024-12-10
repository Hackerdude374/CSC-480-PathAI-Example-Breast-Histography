import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionMapVisualizer:
    """Component for visualizing attention heatmaps"""
    def __init__(self, colorscale: str = "Viridis"):
        self.colorscale = colorscale

    def plot_attention_heatmap(
        self,
        original_image: Image.Image,
        attention_weights: np.ndarray,
        overlay_alpha: float = 0.5
    ):
        """Display attention heatmap overlaid on original image"""
        # Create columns for side-by-side view
        col1, col2, col3 = st.columns(3)
        
        # Original image
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
        
        # Attention heatmap
        with col2:
            st.subheader("Attention Heatmap")
            fig = px.imshow(
                attention_weights,
                color_continuous_scale=self.colorscale
            )
            fig.update_layout(
                coloraxis_showscale=True,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Overlaid view
        with col3:
            st.subheader("Overlay View")
            st.image(
                self._create_overlay(
                    original_image,
                    attention_weights,
                    overlay_alpha
                ),
                use_column_width=True
            )

    def _create_overlay(
        self,
        image: Image.Image,
        attention: np.ndarray,
        alpha: float
    ) -> Image.Image:
        """Create overlay of attention map on image"""
        # Convert attention to heatmap
        plt.figure(figsize=(10, 10))
        heatmap = plt.imshow(attention, cmap='viridis')
        plt.axis('off')
        
        # Save heatmap to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        
        # Open heatmap image and resize to match original
        heatmap_img = Image.open(buf).resize(image.size)
        
        # Blend images
        return Image.blend(image, heatmap_img, alpha)

class PredictionVisualizer:
    """Component for visualizing model predictions"""
    def show_prediction_results(
        self,
        probabilities: List[float],
        confidence_threshold: float = 0.5
    ):
        """Display prediction probabilities and confidence"""
        # Prediction bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=['Benign', 'Malignant'],
                y=probabilities,
                marker_color=['#2ecc71', '#e74c3c'],
                text=[f'{p:.1%}' for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Prediction Probabilities',
            yaxis_title='Probability',
            yaxis_range=[0, 1],
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence indicator
        confidence = max(probabilities)
        st.metric(
            "Prediction Confidence",
            f"{confidence:.1%}",
            delta="Above threshold" if confidence > confidence_threshold else "Below threshold"
        )

class TissueGraphVisualizer:
    """Component for visualizing tissue graph structure"""
    def plot_tissue_graph(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        node_features: Optional[np.ndarray] = None
    ):
        """Display tissue graph visualization"""
        st.subheader("Tissue Graph Structure")
        
        # Create graph figure
        fig = go.Figure()
        
        # Add edges
        for (src, dst) in edges:
            fig.add_trace(go.Scatter(
                x=[nodes[src, 0], nodes[dst, 0]],
                y=[nodes[src, 1], nodes[dst, 1]],
                mode='lines',
                line=dict(color='#888', width=1),
                hoverinfo='none'
            ))
        
        # Add nodes
        node_colors = node_features if node_features is not None else '#1f77b4'
        fig.add_trace(go.Scatter(
            x=nodes[:, 0],
            y=nodes[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=node_colors,
                colorscale='Viridis',
                showscale=bool(node_features is not None)
            ),
            text=[f'Node {i}' for i in range(len(nodes))],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)

class AnalysisReport:
    """Component for generating analysis reports"""
    def create_report(
        self,
        results: Dict,
        original_image: Image.Image,
        include_graphs: bool = True
    ) -> bytes:
        """Generate PDF report with analysis results"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(12, 8))
        
        # Layout
        gs = fig.add_gridspec(2, 2)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Attention heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(results['attention_map'])
        ax2.set_title('Attention Heatmap')
        ax2.axis('off')
        
        # Predictions
        ax3 = fig.add_subplot(gs[1, 0])
        sns.barplot(
            x=['Benign', 'Malignant'],
            y=results['probabilities'],
            ax=ax3
        )
        ax3.set_title('Prediction Probabilities')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='pdf', bbox_inches='tight')
        buf.seek(0)
        
        return buf.getvalue()

def display_batch_results(results: List[Dict]):
    """Display results for batch analysis"""
    # Create summary dataframe
    df = pd.DataFrame([
        {
            'Image': Path(r['image_path']).name,
            'Prediction': 'Malignant' if r['predictions']['predicted_class'] == 1 else 'Benign',
            'Confidence': r['predictions']['confidence'],
            'Processing Time': r['processing_time']
        }
        for r in results
    ])
    
    # Show summary table
    st.dataframe(df)
    
    # Show statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Average Confidence",
            f"{df['Confidence'].mean():.1%}"
        )
        
    with col2:
        st.metric(
            "Average Processing Time",
            f"{df['Processing Time'].mean():.2f}s"
        )
    
    # Distribution plot
    fig = px.histogram(
        df,
        x='Confidence',
        color='Prediction',
        nbins=20,
        title='Confidence Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)