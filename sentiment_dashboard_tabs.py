from fastapi.responses import HTMLResponse
from enhanced_visualisation import create_sentiment_dashboard_plotly, create_emotion_dashboard_plotly
from valence_circumplex_plot import create_circumplex_plot
import pandas as pd
from distortion_detection import detect_distortions
from pydantic import BaseModel

class SentimentSummary(BaseModel):
    emotion: str
    mean: float
    std: float
    max_val: float
    min_val: float

def build_dashboard_tabbed(model_name: str, data, kind: str = "utterance"):
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data

    df = df.copy()
    df["distortions"] = df["utterance"].apply(
        lambda x: ", ".join([d["distortion"] for d in detect_distortions(x)]) or "None"
    )

    html_parts = [f"<h3 style='margin-top:0'>Model: {model_name.title()}</h3>"]

    if kind == "utterance" and "speaker" in df.columns:
        tabs = """
        <style>
        .inner-tab-button {
            margin: 5px;
            padding: 8px 12px;
            background: #3a3a4f;
            border: none;
            color: white;
            cursor: pointer;
            border-radius: 5px;
        }
        .inner-tab-button:hover { background: #5a5a7f; }
        .inner-tab-content { display: none; max-width: 95vw; }
        .inner-tab-content.active { display: block; }
        iframe.plotly-graph-div { width: 100% !important; height: auto !important; }
        </style>
        <script>
        function toggleInnerTab(model, role) {
            document.querySelectorAll(`#tab-${model} .inner-tab-content`).forEach(el => el.classList.remove("active"));
            const activeTab = document.querySelector(`#tab-${model} .inner-tab-content[data-role='" + role + "']`);
            if (activeTab) activeTab.classList.add("active");
        }
        window.addEventListener('resize', () => {
            document.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.Plots.resize(el));
        });
        </script>
        """

        therapist_df = df[df["speaker"] == "Therapist"]
        client_df = df[df["speaker"] == "Client"]

        therapist_figs = create_sentiment_dashboard_plotly(therapist_df, show_text=False)
        client_figs = create_sentiment_dashboard_plotly(client_df, show_text=False)
        circ_therapist = create_circumplex_plot(therapist_df, show_text=False)
        circ_client = create_circumplex_plot(client_df, show_text=False)

        inner_html = f"""
        <div>
            <button class='inner-tab-button' onclick=\"toggleInnerTab('{model_name}', 'Therapist')\">Therapist</button>
            <button class='inner-tab-button' onclick=\"toggleInnerTab('{model_name}', 'Client')\">Client</button>
        </div>
        <div id='tab-{model_name}'>
            <div class='inner-tab-content active' data-role='Therapist'>
                {therapist_figs['scatter'].to_html(full_html=False, include_plotlyjs='cdn')}
                {therapist_figs['valence_hist'].to_html(full_html=False, include_plotlyjs=False)}
                {therapist_figs['arousal_hist'].to_html(full_html=False, include_plotlyjs=False)}
                {circ_therapist.to_html(full_html=False, include_plotlyjs=False)}
            </div>
            <div class='inner-tab-content' data-role='Client'>
                {client_figs['scatter'].to_html(full_html=False, include_plotlyjs=False)}
                {client_figs['valence_hist'].to_html(full_html=False, include_plotlyjs=False)}
                {client_figs['arousal_hist'].to_html(full_html=False, include_plotlyjs=False)}
                {circ_client.to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </div>
        """
        html_parts.append(tabs + inner_html)

    elif kind == "utterance":
        figs = create_sentiment_dashboard_plotly(df, show_text=False)
        circ = create_circumplex_plot(df, show_text=False)
        html_parts += [
            figs['scatter'].to_html(full_html=False, include_plotlyjs='cdn'),
            figs['valence_hist'].to_html(full_html=False, include_plotlyjs=False),
            figs['arousal_hist'].to_html(full_html=False, include_plotlyjs=False),
            circ.to_html(full_html=False, include_plotlyjs=False)
        ]

    elif kind == "summary":
        df = pd.DataFrame([s.__dict__ if isinstance(s, SentimentSummary) else s for s in data])
        figs = create_emotion_dashboard_plotly(df)
        html_parts += [
            figs['box'].to_html(full_html=False, include_plotlyjs='cdn'),
            figs['mean_std'].to_html(full_html=False, include_plotlyjs=False),
            figs['range_bar'].to_html(full_html=False, include_plotlyjs=False)
        ]

    return HTMLResponse(content=f"""
    <div style='background:#0d0c1d;padding:20px;color:white;max-width:95vw;'>
        {''.join(html_parts)}
    </div>
    """)