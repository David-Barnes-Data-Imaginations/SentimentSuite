import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import os
import torch
import matplotlib

matplotlib.use('Agg')
from transformers import pipeline
import pandas as pd
from pydantic import BaseModel
from typing import Tuple, Dict, List
import re
import io
from typing import Any, Optional
from datetime import datetime
from enhanced_visualisation import create_sentiment_dashboard_plotly, create_emotion_dashboard_plotly
# from valence_circumplex_plot import create_circumplex_plot
from sentiment_dashboard_tabs import build_dashboard_tabbed
from circumplex_plot import create_circumplex_plot
from distortion_detection import detect_distortions
from fastapi.responses import HTMLResponse

torch.set_float32_matmul_precision('high')


# Updated the Sentiment2D class with
# more emotions and patterns
class Sentiment2D:
    def __init__(self):
        """Initialize the sentiment analyzer with expanded emotion keywords and their values"""
        self.emotion_map = {
            # Basic emotions
            'happy': (0.8, 0.5),
            'sad': (-0.6, -0.4),
            'angry': (-0.6, 0.8),
            'calm': (0.3, -0.6),
            'excited': (0.5, 0.8),
            'nervous': (-0.3, 0.7),
            'peaceful': (0.4, -0.7),
            'gloomy': (-0.5, -0.5),
            # Additional emotions and phrases to fit the forge
            'welcome': (0.6, 0.2),
            'problem': (-0.4, 0.3),
            'inform': (0.1, -0.2),
            'await': (0.0, 0.3),
            'service': (0.4, 0.0),
            'expletive': (-0.5, 0.6),
            'good': (0.7, 0.3),
            'bad': (-0.7, 0.3),
            'great': (0.8, 0.4),
            'terrible': (-0.8, 0.5),
            'wonderful': (0.9, 0.5),
            'awful': (-0.8, 0.4),
            'pleasant': (0.6, -0.2),
            'unpleasant': (-0.6, 0.2),
            'system': (0.0, -0.3),
            'leave': (-0.2, 0.1)
        }

        # Enhanced pattern matching
        self.patterns = {}
        for emotion in self.emotion_map:
            # Create patterns that match word boundaries and handle potential plurals
            pattern = r'\b' + emotion + r'(?:s|es|ing|ed)?\b'
            self.patterns[emotion] = re.compile(pattern, re.IGNORECASE)

    def get_utterance_class_scores(self, utterance: str) -> Dict[str, float]:
        """Calculate emotion scores with improved matching
        :type utterance: str
        """
        scores = {}
        utterance.lower().split()

        # Initialize all emotions with a small baseline value
        for emotion in self.emotion_map:
            scores[emotion] = 0.01  # Small baseline to avoid complete neutrality

        for emotion, pattern in self.patterns.items():
            # Count occurrences and weight them
            count = len(pattern.findall(utterance))
            if count > 0:
                scores[emotion] = count * 0.5  # Weight the matches

        # Normalize scores
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def get_utterance_valence_arousal(self, utterance: str) -> Tuple[float, float]:
        """Calculate valence and arousal with improved weighting"""
        scores = self.get_utterance_class_scores(utterance)

        valence = 0.0
        arousal = 0.0
        total_weight = 0.0

        for emotion, score in scores.items():
            v, a = self.emotion_map[emotion]
            # Apply score as weight
            valence += v * score
            arousal += a * score
            total_weight += score

        # Normalize and ensure non-zero output
        if total_weight > 0:
            valence = valence / total_weight
            arousal = arousal / total_weight
        else:
            valence = 0.0
            arousal = 0.0

        return (valence, arousal)

    def __call__(self, utterance: str) -> Tuple[float, float]:
        """Process the utterance and return valence-arousal pair"""
        return self.get_utterance_valence_arousal(utterance)


# Set environment variable before importing torch
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# Initialize FastAPI app
app = FastAPI()

# Initialize Sentiment2D
sentiment2d = Sentiment2D()

class SentimentSummary(BaseModel):
    emotion: str
    mean: float
    std: float
    max_val: float
    min_val: float

def build_dashboard_tabbed(model_name: str, data, kind: str = "utterance"):
    if kind == "utterance":
        df = pd.DataFrame(data)
        df["distortions"] = df["utterance"].apply(
            lambda x: ", ".join([d["distortion"] for d in detect_distortions(x)]) or "None"
        )

        main_figs = create_sentiment_dashboard_plotly(df)
        circ_fig = create_circumplex_plot(df)

        html_parts = [
            f"<h3>Model: {model_name}</h3>",
            f"<p><strong>Distortions Detected:</strong><br><pre style='color:#ccc'>{df[['utterance', 'distortions']].to_string(index=False)}</pre></p>",
            main_figs['scatter'].to_html(full_html=False, include_plotlyjs='cdn'),
            main_figs['valence_hist'].to_html(full_html=False, include_plotlyjs=False),
            main_figs['arousal_hist'].to_html(full_html=False, include_plotlyjs=False),
            circ_fig.to_html(full_html=False, include_plotlyjs=False)
        ]

    elif kind == "summary":
        df = pd.DataFrame([s.__dict__ if isinstance(s, SentimentSummary) else s for s in data])
        summary_figs = create_emotion_dashboard_plotly(df)
        html_parts = [
            f"<h3>Model: {model_name}</h3>",
            summary_figs['box'].to_html(full_html=False, include_plotlyjs='cdn'),
            summary_figs['mean_std'].to_html(full_html=False, include_plotlyjs=False),
            summary_figs['range_bar'].to_html(full_html=False, include_plotlyjs=False)
        ]
    else:
        html_parts = ["<p>Unsupported data type</p>"]

    return "".join(html_parts)


@app.get("/dashboard_all", response_class=HTMLResponse)
def dashboard_all_models():
    from SentimentSuite import analysis_store
    tabs_html = []

    for model_name, result_data in analysis_store.results.items():
        if not result_data:
            continue
        kind = "utterance" if model_name in ["bart", "nous-hermes"] else "summary"
        tab_html = build_dashboard_tabbed(model_name, result_data, kind)
        tabs_html.append(f'''
            <div class='tab-content' id='{model_name}' style='display:none'>
                <div class="tab-container">
                    {tab_html}
                </div>
            </div>
        ''')

    buttons = "".join([
        f"<button class='tab-button' onclick=\"showTab('{model}')\">{model.title()}</button>"
        for model in analysis_store.results if analysis_store.results[model]
    ])

    return HTMLResponse(content=f"""
        <html>
        <head>
            <title>SentimentSuite Dashboard</title>
            <style>
                body {{ 
                    background:#1a1a1a; 
                    color:white; 
                    font-family:sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                .tab-button {{ 
                    margin:5px; 
                    padding:10px 20px; 
                    background:#2d2d2d; 
                    border:none; 
                    color:white; 
                    cursor:pointer;
                    border-radius: 5px;
                }}
                .tab-button:hover {{ background:#444; }}
                .tab-button.active {{ background:#4a4a8e; }}
                .tab-content {{ 
                    padding: 20px; 
                    background:#0d0c1d; 
                    margin-top: 10px; 
                    border-radius: 10px;
                    width: 100%;
                }}
                .js-plotly-plot {{ 
                    width: 100% !important; 
                    height: 600px !important;
                }}
                pre {{ 
                    white-space: pre-wrap; 
                    word-wrap: break-word;
                    max-height: 300px;
                    overflow-y: auto;
                    background: #2d2d2d;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .dashboard-button {{ 
                    margin-top: 20px; 
                    display: inline-block; 
                    padding: 12px 24px; 
                    background: #2196F3; 
                    color: white; 
                    border: none; 
                    border-radius: 8px; 
                    text-decoration: none; 
                    font-weight: bold; 
                }}
                .dashboard-button:hover {{ background: #1976D2; }}
            </style>
        </head>
        <body>
            <h1>SentimentSuite Dashboard</h1>
            <div class="tab-buttons">{buttons}</div>
            <div style='margin-top:30px;'>
                <a href="/upload-csv" class="dashboard-button">Upload New CSV</a>
            </div>
            <div class="tab-container">{''.join(tabs_html)}</div>

            <script>
                function showTab(id) {{
                    // Update buttons
                    document.querySelectorAll('.tab-button').forEach(btn => {{
                        if (btn.textContent.toLowerCase() === id) {{
                            btn.classList.add('active');
                        }} else {{
                            btn.classList.remove('active');
                        }}
                    }});

                    // Update content
                    document.querySelectorAll('.tab-content').forEach(div => {{
                        div.style.display = 'none';
                    }});
                    const tab = document.getElementById(id);
                    if (tab) {{
                        tab.style.display = 'block';
                        // Trigger Plotly to resize
                        const plots = tab.getElementsByClassName('js-plotly-plot');
                        for (let plot of plots) {{
                            if (window.Plotly) {{
                                Plotly.relayout(plot, {{
                                    'xaxis.autorange': true,
                                    'yaxis.autorange': true,
                                    'width': plot.offsetWidth,
                                    'height': 600
                                }});
                            }}
                        }}
                    }}
                }}

                // Initialize first tab
                window.onload = () => {{
                    const firstButton = document.querySelector('.tab-button');
                    if (firstButton) {{
                        showTab(firstButton.textContent.toLowerCase());
                    }}
                }};

                // Handle window resize
                window.addEventListener('resize', () => {{
                    const activeTab = document.querySelector('.tab-content[style*="block"]');
                    if (activeTab) {{
                        const plots = activeTab.getElementsByClassName('js-plotly-plot');
                        for (let plot of plots) {{
                            if (window.Plotly) {{
                                Plotly.relayout(plot, {{
                                    'width': plot.offsetWidth,
                                    'height': 600
                                }});
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
    """)

# Global classifiers are now available.
classifier = pipeline(
    "text-classification",
    model="./models/modernbert/1",
    top_k=5,
    device=0 if torch.cuda.is_available() else -1
)


class SentimentSummary(BaseModel):
    emotion: str
    mean: float
    std: float
    max_val: float
    min_val: float


# Add this class to store analysis results
class AnalysisResults:
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.timestamp: Optional[datetime] = None


analysis_store = AnalysisResults()
analysis_store.results = {
    'modernbert': [],
    'bart': [],
    'nous-hermes': []
}


# Modify analysis endpoints to store results
@app.post("/analyze/bart")
def analyze_bart(file: UploadFile = File(...)):
    content = file.file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    df.columns = [c.strip().lower() for c in df.columns]
    if "utterance" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain an 'utterance' column")

    speaker_col = "speaker" if "speaker" in df.columns else None
    results = []

    for _, row in df.iterrows():
        utt = row["utterance"]
        speaker = row[speaker_col] if speaker_col else None
        valence, arousal = sentiment2d(utt)
        record = {
            "utterance": utt,
            "valence": round(valence, 3),
                        "arousal": round(arousal, 3),
        }
        if speaker_col:
            record["speaker"] = speaker
        results.append(record)


    # Store the results
    analysis_store.results['bart'] = results
    analysis_store.timestamp = datetime.now()
    return results


def analyze_nous_hermes(file: UploadFile = File(...)):
    content = file.file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    df.columns = [c.strip().lower() for c in df.columns]
    if "utterance" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain an 'utterance' column")

    speaker_col = "speaker" if "speaker" in df.columns else None
    results = []

    utterances = df["utterance"]

    for _, row in df.iterrows():
        utt = row["utterance"]
        speaker = row[speaker_col] if speaker_col else None
        try:
            # First try the Nous-Hermes server
            payload = {
                "prompt": f"Analyze the emotional tone of: '{utt}' and return in format: {{valence: float, arousal: float}}.",
                "temperature": 0.7,
                "max_tokens": 200
            }

            try:
                # Try to connect to Nous-Hermes with a short timeout
                response = requests.post(
                    "http://localhost:1234/v1/completions",
                    json=payload,
                    timeout=1  # 1 second timeout
                )
                response_data = response.json()
                record = {
                "utterance": utt,
                "model": "nous-hermes",
                "raw_output": response_data.get("choices", [{}])[0].get("text", "")
                }
                if speaker_col:
                    record["speaker"] = speaker
                results.append(record)
            except (requests.exceptions.RequestException, KeyError):

                # If Nous-Hermes fails, fallback to our Sentiment2D
                valence, arousal = sentiment2d(utt)
                record = {
                "utterance": utt,
                "model": "sentiment2d-fallback",
                "valence": round(valence, 3),
                "arousal": round(arousal, 3)
                }
                if speaker_col:
                    record["speaker"] = speaker
                results.append(record)
        except Exception as e:
            record = {
            "utterance": utt,
            "model": "error",
            "error": str(e)
        }
            if speaker_col:
                record["speaker"] = speaker
            results.append(record)

    # Store the results before returning
    analysis_store.results['nous-hermes'] = results
    analysis_store.timestamp = datetime.now()
    return results
@app.get("/analyze/bart/file")
def analyze_bart_get(file: UploadFile = File(...)):
    content = file.file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    df.columns = [c.strip().lower() for c in df.columns]
    if "utterance" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain an 'utterance' column")

    speaker_col = "speaker" if "speaker" in df.columns else None
    results = []

    for _, row in df.iterrows():
        utt = row["utterance"]
        speaker = row[speaker_col] if speaker_col else None
        valence, arousal = sentiment2d(utt)
        distortions = detect_distortions(utt)
        record={
            "utterance": utt,
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "distortions": [d["distortion"] for d in distortions],
        }
        if speaker_col:
            record["speaker"] = speaker
        results.append(record)

@app.get("/upload-csv", response_class=HTMLResponse)
async def upload_form():
    return '''
        <html>
            <head>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        background: #1a1a1a;
                        color: white;
                    }
                    .upload-form {
                        border: 2px dashed #ccc;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        background: #2d2d2d;
                    }
                    .submit-btn {
                        margin-top: 10px;
                        padding: 10px 20px;
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                    }
                    .method-select {
                        margin: 10px 0;
                        padding: 5px;
                        border-radius: 5px;
                    }
                    #results {
                        margin-top: 20px;
                        padding: 10px;
                        background: #3d3d3d;
                        border-radius: 5px;
                        white-space: pre-wrap;
                        max-height: 300px;
                        overflow-y: auto;
                    }
                    .view-dashboard {
                        display: inline-block;
                        margin: 10px 0;
                        padding: 10px 20px;
                        background-color: #2196F3;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                    }
                    .button-container {
                        margin: 20px 0;
                    }
                </style>
            </head>
            <body>
                <h2>Upload CSV File for Sentiment Analysis</h2>
                <div class="upload-form">
                    <form id="uploadForm" enctype="multipart/form-data">
                        <select name="method" class="method-select">
                            <option value="bart">BART Analysis</option>
                            <option value="nous-hermes">Nous-Hermes Analysis</option>
                            <option value="modernbert">ModernBERT Analysis</option>
                        </select>
                        <br>
                        <input name="file" type="file" accept=".csv">
                        <br>
                        <div class="button-container">
                            <button type="submit" class="submit-btn">Upload and Analyze</button>
                            <a href="/dashboard_all" class="view-dashboard">View Dashboard</a>
                        </div>
                    </form>
                    <div id="results"></div>
                </div>

                <script>
                    document.getElementById('uploadForm').onsubmit = async (e) => {
                        e.preventDefault();
                        const formData = new FormData(e.target);
                        const method = formData.get('method');
                        let endpoint = '';

                        switch(method) {
                            case 'bart': endpoint = '/analyze/bart'; break;
                            case 'nous-hermes': endpoint = '/analyze/nous-hermes'; break;
                            case 'modernbert': endpoint = '/upload-csv-process'; break;
                        }

                        try {
                            const response = await fetch(endpoint, {
                                method: 'POST',
                                body: formData
                            });
                            const data = await response.json();
                            document.getElementById('results').innerHTML =
                                '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                        } catch (error) {
                            document.getElementById('results').innerHTML =
                                '<p style="color: red;">Error: ' + error.message + '</p>';
                        }
                    };
                </script>
            </body>
        </html>
    '''

@app.post("/upload-csv-process", response_model=List[SentimentSummary])
async def upload_csv_process(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        all_scores = {}
        for row in df["utterance"]:
            try:
                outputs = classifier(row)[0]
                for item in outputs:
                    label = item["label"]
                    score = item["score"]
                    all_scores.setdefault(label, []).append(score)
            except Exception as e:
                print(f"Error processing row: {row}, Error: {str(e)}")
                continue

        summary = []
        for emotion, values in all_scores.items():
            if values:
                series = pd.Series(values)
                try:
                    mean_val = float(series.mean())
                    std_val = float(series.std())
                    max_val = float(series.max())
                    min_val = float(series.min())
                    if all(abs(x) < 1e308 for x in [mean_val, std_val, max_val, min_val]):
                        summary.append(SentimentSummary(
                            emotion=emotion,
                            mean=mean_val,
                            std=std_val,
                            max_val=max_val,
                            min_val=min_val
                        ))
                except Exception as e:
                    print(f"Error calculating stats for emotion {emotion}: {str(e)}")
                    continue

        if not summary:
            return []

        analysis_store.results['modernbert'] = summary
        analysis_store.timestamp = datetime.now()
        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_sentiment_dashboard(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame([
            {
                'utterance': item['utterance'],
                'valence': item.get('valence', 0),
                'arousal': item.get('arousal', 0)
            } for item in data if 'utterance' in item
        ])
    else:
        df = data

    # 1. Valence-Arousal Scatter Plot
    ax1 = fig.add_subplot(gs[0, :])
    scatter = ax1.scatter(df['valence'], df['arousal'],
                          c=np.arange(len(df)), cmap='viridis',
                          s=100)
    ax1.set_title('Valence-Arousal Space')
    ax1.set_xlabel('Valence')
    ax1.set_ylabel('Arousal')

    # Set specific axis limits
    ax1.set_xlim(-0.37, 0.28)
    ax1.set_ylim(df['arousal'].min() - 0.1, df['arousal'].max() + 0.1)

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Utter')

    # Add tooltips with smaller font and adjusted position
    for i, txt in enumerate(df['utterance']):
        shortened_text = txt[:20] + '...' if len(txt) > 20 else txt
        ax1.annotate(shortened_text,
                     (df['valence'].iloc[i], df['arousal'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8,  # Smaller font size
                     alpha=0.8,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # 2. Valence Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    sns.histplot(data=df, x='valence', kde=True, ax=ax2)
    ax2.set_title('Valence Distribution')

    # 3. Arousal Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    sns.histplot(data=df, x='arousal', kde=True, ax=ax3)
    ax3.set_title('Arousal Distribution')

    plt.tight_layout()
    return fig


def create_emotion_dashboard(data):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame([
            {
                'emotion': item.emotion,
                'mean': item.mean,
                'std': item.std,
                'max_val': item.max_val,
                'min_val': item.min_val
            } for item in data
        ])
    else:
        df = data

    # 1. Boxplot of emotion statistics
    ax1 = fig.add_subplot(gs[0, :])
    df_melted = pd.melt(df, id_vars=['emotion'],
                        value_vars=['mean', 'std', 'max_val', 'min_val'])
    sns.boxplot(data=df_melted, x='emotion', y='value',
                hue='variable', ax=ax1)
    ax1.set_title('Distribution of Emotion Statistics')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # 2. Mean vs Std scatter with adjusted legend
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = sns.scatterplot(data=df, x='mean', y='std', ax=ax2,
                              s=100, hue='emotion')
    # Move legend outside the plot
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax2.set_title('Mean vs Standard Deviation')

    # 3. Range plot
    ax3 = fig.add_subplot(gs[1, 1])
    df['range'] = df['max_val'] - df['min_val']
    sns.barplot(data=df, x='emotion', y='range', ax=ax3)
    ax3.set_title('Emotion Range (Max - Min)')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

    plt.tight_layout()
    return fig


# Update the dashboard endpoint
@app.get("/dashboard/{analysis_type}")
async def get_dashboard(analysis_type: str):
    if analysis_type not in analysis_store.results:
        raise HTTPException(status_code=404,
                            detail=f"No {analysis_type} analysis results found. Please run analysis first.")

    # Get the latest analysis results
    if analysis_type == "modernbert":
        figs = create_emotion_dashboard_plotly(analysis_store.results[analysis_type])
        html_parts = [
            figs["box"].to_html(full_html=False, include_plotlyjs="cdn"),
            figs["mean_std"].to_html(full_html=False, include_plotlyjs=False),
            figs["range_bar"].to_html(full_html=False, include_plotlyjs=False),
        ]
    else:  # bart or nous-hermes
        figs = create_sentiment_dashboard_plotly(analysis_store.results[analysis_type])
        html_parts = [
            figs["scatter"].to_html(full_html=False, include_plotlyjs="cdn"),
            figs["valence_hist"].to_html(full_html=False, include_plotlyjs=False),
            figs["arousal_hist"].to_html(full_html=False, include_plotlyjs=False),
        ]

        # Convert plot to base64 string
        """ buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()"""


        # Create HTML with timestamp
        timestamp_str = analysis_store.timestamp.strftime("%Y-%m-%d %H:%M:%S") if analysis_store.timestamp else "Unknown"
        html_content = f'''
            <html>
                <head>
                    <title>Sentiment Analysis Dashboard</title>
                    <style>
                        body {{ 
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background: #1a1a1a;
                            color: white;
                        }}
                        .dashboard {{
                            max-width: 1200px;
                            margin: 0 auto;
                            background: #2d2d2d;
                            padding: 20px;
                            border-radius: 10px;
                        }}
                        img {{
                            width: 100%;
                            height: auto;
                        }}
                        .timestamp {{
                            color: #888;
                            font-size: 0.8em;
                            margin-top: 10px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="dashboard">
                        <h1>{analysis_type.title()} Analysis Dashboard</h1>
                          {"".join(html_parts)}
                        <div class="timestamp">Last analyzed: {timestamp_str}</div>
                    </div>
                </body>
            </html>
        '''
        return HTMLResponse(content=html_content)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_home():
    return '''
        <html>
            <head>
                <title>Sentiment Analysis Dashboard</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: #1a1a1a;
                        color: white;
                    }
                    .dashboard-links {
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        background: #2d2d2d;
                        border-radius: 10px;
                    }
                    .dashboard-link {
                        display: block;
                        margin: 10px 0;
                        padding: 10px;
                        background: #3d3d3d;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                    }
                    .dashboard-link:hover {
                        background: #4d4d4d;
                    }
                </style>
            </head>
            <body>
                <div class="dashboard-links">
                    <h1>Sentiment Analysis Dashboards</h1>
                    <a href="/upload-csv" class="dashboard-link">Upload New Data</a>
                    <a href="/dashboard/modernbert" class="dashboard-link">ModernBERT Dashboard</a>
                    <a href="/dashboard/bart" class="dashboard-link">BART Dashboard</a>
                    <a href="/dashboard/nous-hermes" class="dashboard-link">Nous-Hermes Dashboard</a>
                </div>
            </body>
        </html>
    '''