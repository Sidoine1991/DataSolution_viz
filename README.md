# DataSolution Viz — Tableau de bord CCR-B

Application Streamlit pour visualiser et analyser les données de collecte KoboToolbox du **Conseil de Concertation des Riziculteurs du Bénin (CCR-B)**.

## Fonctionnalités

- Chargement des données depuis l'API KoboToolbox ou un fichier Excel local
- Visualisations interactives (Plotly)
- Statistiques agrégées et analyses (PCA, clustering, régression)
- Export CSV, Excel et rapport HTML
- Chat IA avec Google Gemini

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
├── dashboard_app.py       # Application principale
├── report_template.html   # Template du rapport HTML
├── PP CCRB.png            # Logo CCR-B
├── requirements.txt       # Dépendances Python
├── render.yaml            # Configuration Render
├── .env.example           # Modèle de variables d'environnement
├── .gitignore
├── .streamlit/
│   └── config.toml
└── README.md
```

## Variables d'environnement

| Variable         | Obligatoire | Description                          |
|------------------|-------------|--------------------------------------|
| `GOOGLE_API_KEY` | Oui         | Clé API Google Gemini pour le chat   |
| `GEMINI_MODEL`   | Non         | Modèle Gemini (défaut : gemini-1.5-flash) |

## Licence

MIT
