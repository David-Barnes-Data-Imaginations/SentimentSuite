
import plotly.graph_objects as go
import pandas as pd
from emotion_mapping import modernbert_va_map


def create_circumplex_plot(data: pd.DataFrame, palette: list[str] = None, show_text: bool = True):
    """
    Create a circular Russell-style Circumplex plot with valence/arousal points.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns: 'utterance', 'valence', 'arousal' OR 'emotion'.
    palette : list of str, optional
        Color palette for the points.
    show_text : bool, optional
        Whether to show emotion label directly on the plot. Defaults to True.

    Returns
    -------
    go.Figure
        The circular Plotly figure.
    """
    import plotly.graph_objects as go
    import numpy as np

    palette = palette or ["#FF37A6", "#8E57FF", "#00B7FF", "#34D399", "#F5A623"]
    df = data.copy().reset_index(drop=True)

    # Inject valence/arousal if they're missing but we have emotion labels
    if "valence" not in df.columns and "emotion" in df.columns:
        df["valence"] = df["emotion"].apply(lambda e: modernbert_va_map.get(e, (0.0, 0.0))[0])
        df["arousal"] = df["emotion"].apply(lambda e: modernbert_va_map.get(e, (0.0, 0.0))[1])

    df["idx"] = df.index

    # Create the circumplex background grid
    theta = np.linspace(0, 2 * np.pi, 500)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    fig = go.Figure()

    # Draw main circle
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='white', dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Add axes (cross)
    fig.add_trace(
        go.Scatter(x=[-1, 1], y=[0, 0], mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False))
    fig.add_trace(
        go.Scatter(x=[0, 0], y=[-1, 1], mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False))

    # Now plot the actual data
    fig.add_trace(go.Scatter(
        x=df["valence"],
        y=df["arousal"],
        mode='markers+text' if show_text else 'markers',
        text=df["emotion"] if show_text and "emotion" in df.columns else None,
        hovertext=df["utterance"] if "utterance" in df.columns else None,
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            size=9,
            color=df["idx"],
            colorscale=palette,
            line=dict(width=1, color='white')
        ),
        hovertemplate="%{hovertext}<br>Valence: %{x}<br>Arousal: %{y}",
        showlegend=False
    ))

    fig.update_layout(
        title="Russell Circumplex View of Valence–Arousal Space",
        template="plotly_dark",
        plot_bgcolor="#0d0c1d",
        paper_bgcolor="#0d0c1d",
        font=dict(color="#F5F5F5"),
        xaxis=dict(range=[-1.1, 1.1], zeroline=False, showgrid=False, title='Valence'),
        yaxis=dict(range=[-1.1, 1.1], zeroline=False, showgrid=False, title='Arousal'),
        height=600
    )

    return fig
