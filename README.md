# SentimentSuite: Modern Sentiment Analysis with Valence–Arousal Modelling
SentimentSuite is an experimental sentiment analysis tool built on top of large‑language models (LLMs) and modern visualisation libraries.
It aims to move beyond the dated, lexicon‑based approaches commonly encountered in many data‑science courses (for example, NLTK’s Vader or Punkt tokenisers) and instead leverage models such as BART and Nous‑Hermes to produce richer, more nuanced emotional insights. At its core, SentimentSuite organises emotions according to Russell’s Circumplex Model, also known as the Valence–Arousal model.

## Why a Valence–Arousal Model?

Traditional sentiment‑analysis systems classify text into a handful of discrete labels—often “positive”, “negative” or “neutral”. While simple to understand, this binary view fails to capture the subtleties of human emotion. Russell’s Circumplex places emotions on a two‑dimensional plane defined by:
- Valence – How pleasant or unpleasant an emotion is (ranging from negative to positive).
- Arousal – How stimulating or activating the emotion is (ranging from calm to excited).
This framework allows the positioning of sentiments like excited, melancholy or calm relative to one another instead of dropping them into blunt categories. 
For example, the tool computes a valence–arousal pair for each utterance using an expanded list of emotion keywords and patterns. Those coordinates are then plotted on a two‑dimensional plane revealing clusters and trends that you cannot see with a unidimensional sentiment score. Distributions of valence and arousal are shown as histograms and more advanced statistics (mean, standard deviation, range) are summarised across emotions.
Features
- LLM‑powered analysis – SentimentSuite can work with modern transformer models (BART and custom ones) to derive rich emotional embeddings for each utterance.
- Enhanced visualisation – This version ships with interactive dashboards built using Plotly. Scatter plots show how utterances distribute across the valence–arousal plane, while histograms and box plots summarise distributions. A cyber‑punk colour palette (neon pinks, violets and blues) gives the charts a modern, vibrant feel.
- API‑driven architecture – Built on FastAPI, the project exposes endpoints for uploading CSV files, invoking different models, and viewing interactive dashboards.
- Modular design – Visualisation functions are separated into their own module (enhanced_visualisation.py) so they can be integrated into larger systems or reused elsewhere.

## Other Frameworks:
Cognitive Distortions (from CBT)

These are biased thought patterns that skew our perception of reality. Examples:

Catastrophising: “Everything is going to fall apart.”
Overgeneralising: “I failed once, so I’ll always fail.”
Mind-reading: “They must think I’m incompetent.”
More to be added

### Getting Started

1. Clone the repository:
       ```
       git clone https://github.com/David-Barnes-Data-Imaginations/SentimentSuite.git
       cd SentimentSuite
       ```
2. Install dependencies. A typical setup requires Python 3.9+ and
packages like `fastapi, transformers`, `pandas` and `plotly`. 

You can install them with:
       ```
       pip install -r requirements.txt
       ```
       If you intend to run the web app with GPU support, ensure torch is installed with the appropriate CUDA version.
3. Run the API:
       ```
       uvicorn SentimentSuite:app --reload --port 8000
       ```

4. Upload your data at . Navigate to `http://localhost:8000/dashboard-all` and upload a CSV file with an utterance column 
- I've added two examples you can use, the utterances of 'Delamain' from 'Cyberpunk 2077'). 
- I've added the 'go-to' Psychology transcript of Carl and Gloria though i need to have my smolagent tidy it for me to the correct CSV format
5. Choose the model you wish to use (e.g., ModernBERT, BART, or Nous‑Hermes).

6. View the dashboard. After analysis completes, click “View Dashboard” (bottom of the page) to see the interactive plots. The figures show each utterance in the valence–arousal space, histograms of valence and arousal, and summary statistics. You can hover over points to see the corresponding text.

## Part of a Larger Vision – The Persona‑Forge
SentimentSuite is a small, self‑contained module within a broader project called The Persona‑Forge. The goal of the Persona‑Forge is to build detailed personality maps by combining knowledge graph technology with psychological frameworks like the Big Five, Myers–Briggs and Russell’s Circumplex. By augmenting language models with these maps, the Forge aims to create highly realistic personalities for:
- Video games – Dynamic NPC’s whose behaviour evolves with the player’s actions.
- Interactive books and chatbots – Characters that feel truly distinct and can react contextually.
- Personalised assistants on mobile devices – Lightweight models that personalise their responses based on your mood and history.
- Security and profiling – Identifying patterns in text that may signal malicious intent (e.g., attackers, fraudsters) or aiding criminal profiling.
One of the figures in the dashboard displays averages and outliers for each emotion. This summary view is deliberately designed to integrate with knowledge graphs: rather than storing every utterance as a node, I’m currently capturing aggregate statistics (mean, standard deviation, range). These statistics can be linked to other entities—such as characters, scenarios or time periods—allowing the Persona‑Forge to reason over emotional trends at scale.

#### Contributing
Contributions are welcome! If you have ideas for new models, improved visualisations or integrations into the Persona‑Forge, feel free to open an issue or submit a pull request. Please ensure that your contribution adheres to the project’s coding standards and includes
appropriate tests/documentation.

#### License
This project is released under the MIT License. See LICENSE for
details.

## TODO:

#### Add in one more Psychology Framework, here's a brief explanation:
 1. Erikson’s Psychosocial Development Model
What it is:
Erikson’s model describes eight life stages, each associated with a core psychological conflict. For example:
Age 0–1: Trust vs. Mistrust
Age 12–18: Identity vs. Role Confusion
Age 40–65: Generativity vs. Stagnation

Why it’s useful:
You can predict behaviour based on stage (e.g. young adults may question purpose, mid-lifers wrestle with contribution vs. burnout).
It enables timeline-based persona evolution, perfect for graph-based NPCs or chatbot personas who “age” or undergo “narrative arcs”.


2. Combine my SmolAgents runner (from the data scientist project to dynamically learn, transcribe and send an annonomised summary to GPT with is extremely good at psychological analysis).

3. Smolagents runner to transcribe and annonomise from recordings
