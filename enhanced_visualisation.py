"""
This module provides enhanced visualization functions for the SentimentSuite project.

Overview
========
The original project (``SentimentSuite.py``) uses ``matplotlib`` and ``seaborn``
to generate static images of analysis results.  While these libraries are
excellent for quick visualisation, they are limited in their ability to
produce interactive, modern‐looking charts suitable for a web portfolio.

This module re‑implements the dashboard creation using Plotly Express.  Plotly
is a high‑level Python wrapper around the D3.js JavaScript library and
supports interactive exploration out of the box.  Charts rendered via Plotly
can be embedded directly into HTML pages using the generated ``.to_html``
method, making them ideal for FastAPI responses or standalone notebooks.

The palette used here is inspired by cyberpunk aesthetics: neon pinks,
violets and blues set against dark backgrounds.  If you prefer a different
palette, feel free to customise the ``CYBERPUNK_PALETTE`` constant.

Functions
---------
``create_sentiment_dashboard_plotly(data: Iterable[dict] | pandas.DataFrame) -> dict``
    Generate interactive figures for valence/arousal analysis.  Returns a
    dictionary with keys ``scatter``, ``valence_hist`` and ``arousal_hist``
    containing Plotly Figure objects.

``create_emotion_dashboard_plotly(data: Iterable[SentimentSummary] | pandas.DataFrame) -> dict``
    Generate interactive figures for emotion summary statistics.  Returns a
    dictionary with keys ``box``, ``mean_std`` and ``range_bar`` containing
    Plotly Figure objects.

Example usage
-------------
```python
from enhanced_visualization import create_sentiment_dashboard_plotly
from fastapi.responses import HTMLResponse

results = [
    {"utterance": "I love this!", "valence": 0.8, "arousal": 0.4},
    {"utterance": "This is terrible", "valence": -0.7, "arousal": 0.6},
    # ... more rows ...
]

figs = create_sentiment_dashboard_plotly(results)

# Build a single HTML page combining the plots.  The first figure includes
# the Plotly JS bundle; subsequent calls can omit it via ``include_plotlyjs=False``.
html_parts = [
    figs['scatter'].to_html(full_html=False, include_plotlyjs='cdn'),
    figs['valence_hist'].to_html(full_html=False, include_plotlyjs=False),
    figs['arousal_hist'].to_html(full_html=False, include_plotlyjs=False),
]

html = "<div style='background:#0d0c1d;padding:20px;color:white'>" + "".join(html_parts) + "</div>"
return HTMLResponse(content=html)
```

Note: Plotly figures can also be converted to PNG or JPEG via ``fig.write_image``
if static exports are still required.  When serving via FastAPI, sending the
interactive HTML is usually preferable.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from typing import Iterable, Dict, Any, List, Union

# Define a cyberpunk-inspired colour palette (neon pinks, violets and blues).
# This palette is used for discrete colour assignments.  Continuous colour
# scales are derived from these colours.
CYBERPUNK_PALETTE: List[str] = [
    "#FF37A6",  # neon pink
    "#8E57FF",  # violet
    "#00B7FF",  # bright cyan
    "#34D399",  # mint green
    "#F5A623",  # amber
]


def _prepare_sentiment_dataframe(data: Union[pd.DataFrame, Iterable[dict]]) -> pd.DataFrame:
    """Ensure input data is in DataFrame form.

    Parameters
    ----------
    data : pandas.DataFrame or iterable of dict
        Data containing at least ``utterance``, ``valence`` and ``arousal`` keys.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``utterance``, ``valence`` and ``arousal``.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy().reset_index(drop=True)
    rows: List[dict] = []
    for item in data:
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected iterable of dicts or DataFrame, got element of type {type(item)!r}"  # noqa: E501
            )
        # default values if keys are missing
        val = item.get("valence", 0.0)
        aro = item.get("arousal", 0.0)
        text = item.get("utterance", "")
        rows.append({"utterance": text, "valence": val, "arousal": aro})
    return pd.DataFrame(rows)


def _prepare_emotion_dataframe(data: Union[pd.DataFrame, Iterable[Any]]) -> pd.DataFrame:
    """Ensure input emotion summary is in DataFrame form.

    Accepts either a DataFrame with the required columns or an iterable of
    objects having ``emotion``, ``mean``, ``std``, ``max_val`` and ``min_val``
    attributes (e.g. SentimentSummary instances).

    Parameters
    ----------
    data : pandas.DataFrame or iterable
        Data containing emotion statistics.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``emotion``, ``mean``, ``std``, ``max_val`` and ``min_val``.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy().reset_index(drop=True)
    rows: List[dict] = []
    for item in data:
        # Attempt to handle both dicts and objects with attributes
        emotion = getattr(item, "emotion", None) or item.get("emotion")  # type: ignore
        mean = getattr(item, "mean", None) or item.get("mean")  # type: ignore
        std = getattr(item, "std", None) or item.get("std")  # type: ignore
        max_val = getattr(item, "max_val", None) or item.get("max_val")  # type: ignore
        min_val = getattr(item, "min_val", None) or item.get("min_val")  # type: ignore
        if emotion is None:
            raise ValueError("Missing emotion attribute or key in item")
        rows.append(
            {
                "emotion": str(emotion),
                "mean": float(mean),
                "std": float(std),
                "max_val": float(max_val),
                "min_val": float(min_val),
            }
        )
    return pd.DataFrame(rows)


def create_sentiment_dashboard_plotly(
    data: Union[pd.DataFrame, Iterable[Dict[str, Any]]],
    *,
    palette: List[str] = CYBERPUNK_PALETTE,
) -> Dict[str, "plotly.graph_objs._figure.Figure"]:
    """Create interactive Plotly figures for valence/arousal analysis.

    Parameters
    ----------
    data : pandas.DataFrame or iterable of dict
        Must contain ``utterance``, ``valence`` and ``arousal`` columns/keys.
    palette : list of str, optional
        A list of hex colour codes defining the discrete palette used for
        chart elements.  Defaults to ``CYBERPUNK_PALETTE``.

    Returns
    -------
    dict
        A dictionary with keys ``scatter``, ``valence_hist`` and
        ``arousal_hist`` mapping to Plotly Figure objects.
    """
    df = _prepare_sentiment_dataframe(data)

    # Create an index column so that colours can be assigned uniquely.
    df = df.reset_index(drop=True)
    df["idx"] = df.index

    # Scatter plot of valence vs arousal with tooltips showing the utterance.
    scatter_fig = px.scatter(
        df,
        x="valence",
        y="arousal",
        text="utterance",
        color="idx",
        color_continuous_scale=palette,
        title="Valence‑Arousal Space",
        labels={"valence": "Valence", "arousal": "Arousal", "idx": "Utterance"},
    )
    scatter_fig.update_traces(
        marker=dict(size=9, line=dict(width=1, color="#FFFFFF")),
        textposition="top center",
        textfont=dict(color="#FFFFFF", size=10),
    )
    scatter_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0d0c1d",
        paper_bgcolor="#0d0c1d",
        font=dict(color="#F5F5F5"),
        coloraxis_showscale=False,
    )

    # Histogram of valence values
    valence_hist = px.histogram(
        df,
        x="valence",
        nbins=max(10, int(len(df) / 4)),
        color_discrete_sequence=[palette[0]],
        title="Valence Distribution",
        labels={"valence": "Valence", "count": "Count"},
    )
    valence_hist.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0d0c1d",
        paper_bgcolor="#0d0c1d",
        font=dict(color="#F5F5F5"),
    )

    # Histogram of arousal values
    arousal_hist = px.histogram(
        df,
        x="arousal",
        nbins=max(10, int(len(df) / 4)),
        color_discrete_sequence=[palette[2] if len(palette) > 2 else palette[0]],
        title="Arousal Distribution",
        labels={"arousal": "Arousal", "count": "Count"},
    )
    arousal_hist.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0d0c1d",
        paper_bgcolor="#0d0c1d",
        font=dict(color="#F5F5F5"),
    )

    return {
        "scatter": scatter_fig,
        "valence_hist": valence_hist,
        "arousal_hist": arousal_hist,
    }


def create_emotion_dashboard_plotly(
    data: Union[pd.DataFrame, Iterable[Any]],
    *,
    palette: List[str] = CYBERPUNK_PALETTE,
) -> Dict[str, "plotly.graph_objs._figure.Figure"]:
    """Create interactive Plotly figures for emotion summary statistics.

    Parameters
    ----------
    data : pandas.DataFrame or iterable of objects/dicts
        Must contain ``emotion``, ``mean``, ``std``, ``max_val`` and ``min_val``.
    palette : list of str, optional
        Colour palette to use for discrete series.  Defaults to
        ``CYBERPUNK_PALETTE``.

    Returns
    -------
    dict
        A dictionary with keys ``box``, ``mean_std`` and ``range_bar``
        mapping to Plotly Figure objects.
    """
    df = _prepare_emotion_dataframe(data)

    # Melt DataFrame for box plot across metrics
    df_melted = df.melt(
        id_vars=["emotion"],
        value_vars=["mean", "std", "max_val", "min_val"],
        var_name="metric",
        value_name="value",
    )

    # Box plot showing distribution of mean/std/max/min values per emotion
    box_fig = px.box(
        df_melted,
        x="emotion",
        y="value",
        color="metric",
        color_discrete_sequence=palette,
        title="Distribution of Emotion Statistics",
        labels={"value": "Value", "emotion": "Emotion", "metric": "Statistic"},
    )
    box_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0d0c1d",
        paper_bgcolor="#0d0c1d",
        font=dict(color="#F5F5F5"),
        boxmode="group",
    )

    # Scatter plot of mean vs std with marker size reflecting range (max_val - min_val)
    df = df.copy()
    df["range"] = df["max_val"] - df["min_val"]
    mean_std_fig = px.scatter(
        df,
        x="mean",
        y="std",
        size="range",
        color="emotion",
        color_discrete_sequence=palette,
        title="Mean vs Standard Deviation (Marker Size = Range)",
        labels={"mean": "Mean", "std": "Standard Deviation", "range": "Range", "emotion": "Emotion"},
    )
    mean_std_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0d0c1d",
        paper_bgcolor="#0d0c1d",
        font=dict(color="#F5F5F5"),
        legend_title_text="Emotion",
    )

    # Bar plot of range per emotion
    range_bar = px.bar(
        df,
        x="emotion",
        y="range",
        color="emotion",
        color_discrete_sequence=palette,
        title="Emotion Range (Max – Min)",
        labels={"range": "Range", "emotion": "Emotion"},
    )
    range_bar.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0d0c1d",
        paper_bgcolor="#0d0c1d",
        font=dict(color="#F5F5F5"),
        showlegend=False,
    )

    return {
        "box": box_fig,
        "mean_std": mean_std_fig,
        "range_bar": range_bar,
    }
