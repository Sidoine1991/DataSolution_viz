# DataSolution Viz

Application Streamlit personnelle pour visualiser et analyser des données multi-sources (KoboToolbox, Excel, CSV, texte).

## Sources de données

| Mode | Description |
|------|-------------|
| **URL API KoboToolbox** | URL d'export prédéfinie ou personnalisée (.xlsx / .csv) |
| **Fichier Excel** | Import `.xlsx` avec détection automatique des en-têtes |
| **Fichier CSV** | Import `.csv` — première ligne = en-têtes |
| **Texte simple** | Coller des données tabulaires (CSV, `;`, tabulation) |

## Déploiement Streamlit Cloud

- **Repository** : `Sidoine1991/DataSolution_viz`
- **Branch** : `main`
- **Main file** : `dashboard_app.py`
- **Secrets** : `GOOGLE_API_KEY` (pour le chat Gemini)

## Fichiers du projet

| Fichier | Rôle |
|---------|------|
| `dashboard_app.py` | Application principale Streamlit |
| `smart_analysis.py` | Moteur d'analyse intelligente (profiling, insights) |
| `ai_interpreter.py` | Interprétation IA multi-fournisseurs |
| `report_builder.py` | Export rapport Word (.docx) |
| `ccc_analysis_tool.py` | Pipeline d'analyse avancée (PCA, clustering, stats) |
| `logo_sky.png` | Logo |
| `requirements.txt` | Dépendances Python |

## Fonctionnalités

- **Sources** : URL KoboToolbox, Excel, CSV, texte tabulaire
- **Analyse intelligente** : profiling auto, insights, dashboard smart
- **Analyses avancées** : tableaux agrégés, graphiques Plotly, stats, SQL naturel, PCA, clustering
- **IA** : Gemini, OpenAI (fallback), interprétation automatique
- **Export** : CSV, Excel, HTML, PDF, Word

## Prérequis

- Python 3.10 ou supérieur
- Clé API Google Gemini (pour le chat IA)

## Installation locale

```bash
git clone https://github.com/VOTRE_ORG/DataSolution_viz.git
cd DataSolution_viz
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Éditez .env et renseignez GOOGLE_API_KEY
streamlit run dashboard_app.py
```

L'application sera accessible sur [http://localhost:8501](http://localhost:8501).

## Déploiement

### GitHub

1. Créez un dépôt GitHub (ex. `DataSolution_viz`)
2. Depuis ce dossier :

```bash
git init
git add .
git commit -m "Initial commit — dashboard CCR-B"
git branch -M main
git remote add origin https://github.com/VOTRE_ORG/DataSolution_viz.git
git push -u origin main
```

> **Important :** ne commitez jamais le fichier `.env` (il est ignoré par `.gitignore`).

### Render.com

1. Connectez votre dépôt GitHub à [Render](https://render.com)
2. Render détecte automatiquement `render.yaml`
3. Ajoutez la variable d'environnement `GOOGLE_API_KEY` dans le tableau de bord Render

### Streamlit Community Cloud

1. Allez sur [share.streamlit.io](https://share.streamlit.io)
2. Connectez le dépôt GitHub
3. Fichier principal : `dashboard_app.py`
4. Ajoutez `GOOGLE_API_KEY` dans **Settings → Secrets**

## Structure du projet

```
DataSolution_viz/
├── dashboard_app.py
├── smart_analysis.py
├── ai_interpreter.py
├── report_builder.py
├── ccc_analysis_tool.py
├── logo_sky.png
├── requirements.txt
├── render.yaml
├── .env.example
└── README.md
```

## Variables d'environnement

| Variable | Obligatoire | Description |
|----------|-------------|-------------|
| `GOOGLE_API_KEY` | Recommandé | Clé API Google Gemini |
| `OPENAI_API_KEY` | Non | Fallback si quota Gemini |
| `KOBOTOOLBOX_TOKEN` | Non | Formulaires Kobo privés |
| `GEMINI_MODEL` | Non | Modèle Gemini (défaut : gemini-1.5-flash) |
| `LOCAL_GEMMA_PATH` | Non | Modèle local (laisser vide sur le cloud) |

> Aucune clé API n'est stockée dans le code — configurez-les via `.env` (local) ou **Secrets** (Streamlit Cloud).

## Licence

MIT
