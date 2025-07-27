from fastapi.responses import HTMLResponse
from enhanced_visualisation import create_sentiment_dashboard_plotly, create_emotion_dashboard_plotly
from valence_circumplex_plot import create_circumplex_plot


def build_dashboard_tabbed(model_name: str, data, kind: str = "utterance"):
    # kind can be 'utterance' or 'summary'
    if kind == "utterance":
        main_figs = create_sentiment_dashboard_plotly(data)
        circ_fig = create_circumplex_plot(pd.DataFrame(data))

        html_parts = [
            main_figs['scatter'].to_html(full_html=False, include_plotlyjs='cdn'),
            main_figs['valence_hist'].to_html(full_html=False, include_plotlyjs=False),
            main_figs['arousal_hist'].to_html(full_html=False, include_plotlyjs=False),
            circ_fig.to_html(full_html=False, include_plotlyjs=False)
        ]
    else:
        summary_figs = create_emotion_dashboard_plotly(data)
        html_parts = [
            summary_figs['box'].to_html(full_html=False, include_plotlyjs='cdn'),
            summary_figs['mean_std'].to_html(full_html=False, include_plotlyjs=False),
            summary_figs['range_bar'].to_html(full_html=False, include_plotlyjs=False)
        ]

    html = f"""
    <div style='background:#0d0c1d;padding:20px;color:white'>
        <h2>{model_name.title()} Analysis</h2>
        {''.join(html_parts)}
    </div>
    """
    return HTMLResponse(content=html)



