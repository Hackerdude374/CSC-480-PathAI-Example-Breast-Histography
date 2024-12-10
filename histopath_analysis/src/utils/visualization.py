import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import cv2
from typing import List, Tuple, Optional
import plotly.graph_objects as go
import io

def create_attention_heatmap(
    original_image: Image.Image,
    attention_weights: torch.Tensor,
    patch_size: int = 50,
    alpha: float = 0.6
) -> np.ndarray:
    """Create attention heatmap overlay on original image"""
    # Convert image to numpy array
    image_np = np.array(original_image)
    
    # Reshape attention weights to match image grid
    grid_size = int(np.sqrt(len(attention_weights)))
    attention_map = attention_weights.reshape(grid_size, grid_size)
    
    # Resize attention map to match image size
    attention_map = cv2.resize(
        attention_map,
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min()
    )
    
    # Create heatmap
    heatmap = cv2.applyColorMap(
        (attention_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(
        image_np,
        1 - alpha,
        heatmap,
        alpha,
        0
    )
    
    return overlay

def plot_prediction_confidence(
    probabilities: np.ndarray,
    class_names: List[str]
) -> go.Figure:
    """Create bar plot of prediction confidence"""
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities,
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f'{p:.2%}' for p in probabilities],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Prediction Confidence',
        xaxis_title='Class',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        showlegend=False
    )
    
    return fig

def visualize_tissue_graph(
    graph: torch.Tensor,
    node_colors: Optional[torch.Tensor] = None,
    edge_weights: Optional[torch.Tensor] = None
) -> go.Figure:
    """Create interactive visualization of tissue graph"""
    # Get node positions
    pos = graph.pos.numpy()
    
    # Create node trace
    node_trace = go.Scatter(
        x=pos[:, 0],
        y=pos[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=node_colors.numpy() if node_colors is not None else '#1f77b4',
            colorscale='Viridis',
            showscale=True if node_colors is not None else False
        ),
        text=[f'Node {i}' for i in range(len(pos))],
        hoverinfo='text'
    )
    
    # Create edge traces
    edge_traces = []
    edges = graph.edge_index.t().numpy()
    
    if edge_weights is not None:
        weights = edge_weights.numpy()
    else:
        weights = np.ones(len(edges))
    
    for (src, dst), weight in zip(edges, weights):
        edge_traces.append(go.Scatter(
            x=[pos[src, 0], pos[dst, 0]],
            y=[pos[src, 1], pos[dst, 1]],
            mode='lines',
            line=dict(
                width=weight * 2,
                color='#888'
            ),
            hoverinfo='none'
        ))
    
    # Combine traces
    fig = go.Figure(data=[*edge_traces, node_trace])
    
    fig.update_layout(
        title='Tissue Graph Structure',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def create_analysis_report(
    image_path: str,
    predictions: dict,
    attention_map: np.ndarray,
    graph_fig: go.Figure
) -> bytes:
    """Create PDF report with analysis results"""
    plt.style.use('seaborn')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    
    # Original image with attention overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(attention_map)
    ax1.set_title('Attention Heatmap')
    ax1.axis('off')
    
    # Prediction confidence
    ax2 = fig.add_subplot(gs[0, 1])
    class_names = ['Benign', 'Malignant']
    sns.barplot(
        x=class_names,
        y=predictions['class_probabilities'],
        ax=ax2
    )
    ax2.set_title('Prediction Confidence')
    ax2.set_ylim(0, 1)
    
    # Tissue graph
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title('Tissue Graph Analysis')
    # Convert plotly figure to matplotlib
    graph_img = Image.open(io.BytesIO(graph_fig.to_image(format="png")))
    ax3.imshow(graph_img)
    ax3.axis('off')
    
    # Add metadata
    plt.figtext(
        0.02, 0.02,
        f"Analysis Date: {predictions['metadata']['date']}\n" +
        f"Model Version: {predictions['metadata']['model_version']}",
        fontsize=8
    )
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    
    return buf.getvalue()

def plot_training_metrics(
    metrics: dict,
    save_path: Optional[str] = None
) -> go.Figure:
    """Plot training metrics over time"""
    fig = go.Figure()
    
    # Add traces for each metric
    for metric_name, values in metrics.items():
        fig.add_trace(go.Scatter(
            y=values,
            name=metric_name,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Training Metrics',
        xaxis_title='Epoch',
        yaxis_title='Value',
        hovermode='x'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig