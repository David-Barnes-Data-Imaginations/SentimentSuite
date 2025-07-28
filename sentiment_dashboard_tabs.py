from fastapi.responses import HTMLResponse
from enhanced_visualisation import create_sentiment_dashboard_plotly, create_emotion_dashboard_plotly
from valence_circumplex_plot import create_circumplex_plot
import pandas as pd

# Define CSS styles separately
CSS_STYLES = r"""
<style>
.tabs {
    overflow: hidden;
    border: 1px solid #ccc;
    background-color: #1a1a2e;
}
.tabs button {
    background-color: inherit;
    float: left;
    border: none;
    outline: none;
    cursor: pointer;
    padding: 14px 16px;
    transition: 0.3s;
    color: white;
}
.tabs button:hover {
    background-color: #2a2a4e;
}
.tabs button.active {
    background-color: #4a4a6e;
}
.tabcontent {
    display: none;
    padding: 6px 12px;
    border: 1px solid #ccc;
    border-top: none;
}
</style>
"""

# Define JavaScript code separately
JS_CODE = r"""
<script>
function openSpeaker(evt, speakerName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(speakerName).style.display = "block";
    evt.currentTarget.className += " active";
}
</script>
"""

def build_dashboard_tabbed(model_name: str, data, kind: str = "utterance"):
    # Convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data

    # Split data by speaker if available
    if 'speaker' in df.columns:
        therapist_data = df[df['speaker'] == 'Therapist']
        client_data = df[df['speaker'] == 'Client']
        
        # Create figures for each speaker
        therapist_figs = create_sentiment_dashboard_plotly(therapist_data) if kind == "utterance" else create_emotion_dashboard_plotly(therapist_data)
        client_figs = create_sentiment_dashboard_plotly(client_data) if kind == "utterance" else create_emotion_dashboard_plotly(client_data)
        
        # Create HTML with tabs
        html = f"""
        <div style='background:#0d0c1d;padding:20px;color:white'>
            <h2>{model_name.title()} Analysis</h2>
            
            <div class="tabs">
                <button class="tablinks active" onclick="openSpeaker(event, 'Therapist')">Therapist</button>
                <button class="tablinks" onclick="openSpeaker(event, 'Client')">Client</button>
            </div>
            
            <div id="Therapist" class="tabcontent" style="display:block;">
                <h3>Therapist Analysis</h3>
                {therapist_figs['scatter'].to_html(full_html=False, include_plotlyjs='cdn') if kind == "utterance" else therapist_figs['box'].to_html(full_html=False, include_plotlyjs='cdn')}
                {therapist_figs['valence_hist'].to_html(full_html=False, include_plotlyjs=False) if kind == "utterance" else therapist_figs['mean_std'].to_html(full_html=False, include_plotlyjs=False)}
                {therapist_figs['arousal_hist'].to_html(full_html=False, include_plotlyjs=False) if kind == "utterance" else therapist_figs['range_bar'].to_html(full_html=False, include_plotlyjs=False)}
            </div>
            
            <div id="Client" class="tabcontent" style="display:none;">
                <h3>Client Analysis</h3>
                {client_figs['scatter'].to_html(full_html=False, include_plotlyjs=False) if kind == "utterance" else client_figs['box'].to_html(full_html=False, include_plotlyjs=False)}
                {client_figs['valence_hist'].to_html(full_html=False, include_plotlyjs=False) if kind == "utterance" else client_figs['mean_std'].to_html(full_html=False, include_plotlyjs=False)}
                {client_figs['arousal_hist'].to_html(full_html=False, include_plotlyjs=False) if kind == "utterance" else client_figs['range_bar'].to_html(full_html=False, include_plotlyjs=False)}
            </div>
            
            {CSS_STYLES}
            {JS_CODE}
        </div>
        """
    else:
        # Fall back to original implementation if no speaker column
        if kind == "utterance":
            main_figs = create_sentiment_dashboard_plotly(df)
            html_parts = [
                main_figs['scatter'].to_html(full_html=False, include_plotlyjs='cdn'),
                main_figs['valence_hist'].to_html(full_html=False, include_plotlyjs=False),
                main_figs['arousal_hist'].to_html(full_html=False, include_plotlyjs=False)
            ]
        else:
            summary_figs = create_emotion_dashboard_plotly(df)
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