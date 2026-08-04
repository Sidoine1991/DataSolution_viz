import streamlit as st
import pandas as pd
import requests
import io
import re
from io import StringIO
import plotly.express as px
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import scipy.stats as stats
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration produit ---
APP_NAME = "DataSolution Viz"
APP_TAGLINE = "Solution digitale d'analyse et de visualisation de données"
LOGO_PATH = "logo_sky.png"
LOGO_FALLBACK = r"C:\Users\USER\OneDrive\Images\logo_sky.png"

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

# URLs KoboToolbox prédéfinies (optionnelles)
urls = {
    "Enquête 1: Diagnostic rapide des coopératives": "https://eu.kobotoolbox.org/api/v2/assets/aX8mpWRZaVBomEs3jZ5ULR/export-settings/esp68CyMYKKrHVSPDhwGc2X/data.xlsx",
    "Collecte 2: Unité de démonstration/application": "https://eu.kobotoolbox.org/api/v2/assets/aqKsjwyNuGzbwxWHkUeRjj/export-settings/escMDPrVnDBqMdzxBW8Cn3C/data.xlsx",
    "Collecte 3: Diagnostic des coopératives (PDA 4)": "https://eu.kobotoolbox.org/api/v2/assets/adHBEPncaoH7ShGzCaRZoo/export-settings/esnTMAAYFxPGqwCcwuPpKyA/data.xlsx",
}


def get_logo_path():
    if os.path.exists(LOGO_PATH):
        return LOGO_PATH
    if os.path.exists(LOGO_FALLBACK):
        return LOGO_FALLBACK
    return None


def normalize_dataframe(df):
    """Normalise les en-têtes et prépare le DataFrame pour l'analyse."""
    if df is None or df.empty:
        return None

    new_columns = []
    seen = set()
    for col in df.columns:
        cleaned = str(col).split("/")[-1].strip()
        cleaned = re.sub(r"[^\w\s-]", "", cleaned).strip()
        cleaned = re.sub(r"[-\s]+", "_", cleaned).lower() or "colonne"
        base = cleaned
        counter = 1
        while cleaned in seen:
            cleaned = f"{base}_{counter}"
            counter += 1
        seen.add(cleaned)
        new_columns.append(cleaned)
    df = df.copy()
    df.columns = new_columns

    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed")], errors="ignore")

    if "_index" not in df.columns:
        for candidate in ("_index", "index", "id", "formhub_uuid"):
            if candidate in df.columns:
                df["_index"] = df[candidate].astype(str)
                break
        else:
            df["_index"] = df.index.astype(str)

    return df


def store_loaded_data(df, source_label=""):
    if df is not None:
        st.session_state.dataframe_to_export = df
        if source_label:
            st.session_state.current_source_label = source_label
    return df


def detect_text_separator(text):
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    if "\t" in first_line:
        return "\t"
    if ";" in first_line and "," not in first_line:
        return ";"
    return ","


@st.cache_data(ttl=600)
def load_data_from_url(url):
    """Charge les données depuis une URL KoboToolbox (Excel)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        if url.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(response.content))
        else:
            df = pd.read_excel(io.BytesIO(response.content), engine="openpyxl")
        return normalize_dataframe(df)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur réseau lors du chargement depuis {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Erreur lors du traitement des données depuis {url}: {e}")
        return None


def load_data_from_excel(uploaded_file):
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    return normalize_dataframe(df)


def load_data_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return normalize_dataframe(df)


def load_data_from_text(raw_text, separator=None):
    if not raw_text or not raw_text.strip():
        st.warning("Veuillez coller des données tabulaires (première ligne = en-têtes).")
        return None
    sep = separator or detect_text_separator(raw_text)
    df = pd.read_csv(StringIO(raw_text.strip()), sep=sep, header=0)
    return normalize_dataframe(df)

# Charger le template HTML
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template("report_template.html")

def generate_html_report(data, num_submissions, columns, tables_html="", charts_html=""):
    # Rendre le template avec les données
    last_sync = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_content = template.render(
        last_sync=last_sync,
        num_submissions=num_submissions,
        columns=columns,
        tables=tables_html,
        charts=charts_html
    )
    return html_content

logo_path = get_logo_path()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if logo_path:
        st.image(logo_path, width=120)
    else:
        st.warning("Logo non trouvé (logo_sky.png)")
st.markdown(f"<h1 style='text-align: center;'>{APP_NAME}</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center;'>{APP_TAGLINE}</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='border: 1px solid #ccc; padding: 15px; border-radius: 10px; margin-bottom: 20px; color: #000;'>
        <p>Plateforme d'analyse de données multi-sources : API KoboToolbox, Excel, CSV ou texte tabulaire.
        À l'import, l'application détecte automatiquement les en-têtes et structure le DataFrame pour l'analyse.</p>
        <p>Produit DataSolution — analyse, visualisation et export de vos collectes.</p>
    </div>
    """, unsafe_allow_html=True
)

# Display last synchronization time (approximated by data load time)
st.write(f"Dernière synchronisation des données : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs for the application, user manual, and chat
app_tab, manual_tab, chat_tab = st.tabs(["Application", "Manuel d'utilisation", "Chat avec Gemini"])

with app_tab:
    # Add a sidebar for controls
    st.sidebar.header("Configuration")

    data_source_mode = st.sidebar.radio(
        "Sélectionnez la source de données :",
        ("URL API KoboToolbox", "Fichier Excel", "Fichier CSV", "Texte simple"),
    )

    selected_collection_name = None
    uploaded_file = None
    data = None
    source_label = st.session_state.get("current_source_label", "Données")

    if data_source_mode == "URL API KoboToolbox":
        url_mode = st.sidebar.radio(
            "Type d'URL :",
            ("Collecte prédéfinie", "URL personnalisée"),
        )
        if url_mode == "Collecte prédéfinie":
            selected_collection_name = st.sidebar.selectbox(
                "Sélectionnez une collecte :",
                list(urls.keys()),
            )
            selected_url = urls.get(selected_collection_name, "")
        else:
            selected_collection_name = "URL personnalisée"
            selected_url = st.sidebar.text_input(
                "URL d'export KoboToolbox (.xlsx ou .csv)",
                placeholder="https://eu.kobotoolbox.org/api/v2/assets/.../data.xlsx",
            )

        st.header(selected_collection_name or "Chargement API KoboToolbox")
        if selected_url:
            data = load_data_from_url(selected_url)
            if data is not None:
                label = selected_collection_name or f"URL: {selected_url[:50]}..."
                store_loaded_data(data, label)
                st.success(f"Données chargées : {len(data)} lignes, {len(data.columns)} colonnes")
                source_label = st.session_state.get("current_source_label", label)

    elif data_source_mode == "Fichier Excel":
        st.header("Charger un fichier Excel")
        uploaded_file = st.file_uploader("Déposez votre fichier Excel (.xlsx)", type=["xlsx"])
        if uploaded_file is not None:
            try:
                data = load_data_from_excel(uploaded_file)
                store_loaded_data(data, f"Excel: {uploaded_file.name}")
                st.success("Fichier Excel chargé — en-têtes détectés automatiquement.")
                source_label = st.session_state.get("current_source_label", uploaded_file.name)
            except Exception as e:
                st.error(f"Erreur lors du chargement Excel : {e}")

    elif data_source_mode == "Fichier CSV":
        st.header("Charger un fichier CSV")
        uploaded_file = st.file_uploader("Déposez votre fichier CSV (.csv)", type=["csv"])
        if uploaded_file is not None:
            try:
                data = load_data_from_csv(uploaded_file)
                store_loaded_data(data, f"CSV: {uploaded_file.name}")
                st.success("Fichier CSV chargé — en-têtes détectés automatiquement.")
                source_label = st.session_state.get("current_source_label", uploaded_file.name)
            except Exception as e:
                st.error(f"Erreur lors du chargement CSV : {e}")

    elif data_source_mode == "Texte simple":
        st.header("Coller des données tabulaires")
        st.caption("Collez vos données : la première ligne sera utilisée comme en-têtes de colonnes.")
        raw_text = st.text_area(
            "Données (CSV, point-virgule ou tabulation)",
            height=200,
            placeholder="nom,age,ville\nAlice,30,Cotonou\nBob,25,Parakou",
        )
        text_sep = st.selectbox("Séparateur (auto si vide)", ["Auto", ",", ";", "Tabulation"], index=0)
        if st.button("Charger le texte", type="primary"):
            sep_map = {"Auto": None, ",": ",", ";": ";", "Tabulation": "\t"}
            try:
                data = load_data_from_text(raw_text, separator=sep_map[text_sep])
                if data is not None:
                    store_loaded_data(data, "Texte simple")
                    st.success(f"DataFrame créé : {len(data)} lignes, {len(data.columns)} colonnes")
                    source_label = st.session_state.get("current_source_label", "Texte simple")
            except Exception as e:
                st.error(f"Erreur lors de la lecture du texte : {e}")

    # Add a manual refresh button
    if st.sidebar.button("Actualiser les données"):
        st.cache_data.clear()
        st.rerun()

    # Add column renaming functionality
    st.sidebar.subheader("Renommer les colonnes")
    if data is not None:
        col_to_rename = st.sidebar.selectbox(
            "Sélectionnez la colonne à renommer :",
            data.columns.tolist(),
            key=f"rename_select_{selected_collection_name if selected_collection_name else 'local'}"
        )
        new_name = st.sidebar.text_input(
            f"Nouveau nom pour '{col_to_rename}':",
            key=f"rename_text_{selected_collection_name if selected_collection_name else 'local'}"
        )
        if st.sidebar.button("Appliquer le renommage"):
            if col_to_rename and new_name:
                data.rename(columns={col_to_rename: new_name}, inplace=True)
                st.sidebar.success(f"Colonne '{col_to_rename}' renommée en '{new_name}'")
                st.rerun() # Rerun to update the displayed data and selectboxes
            else:
                st.sidebar.warning("Veuillez sélectionner une colonne et entrer un nouveau nom.")
    else:
        st.sidebar.info("Chargez des données pour renommer les colonnes.")

    # Add export options
    st.sidebar.subheader("Exporter les données")
    # Use the dataframe stored in session state for export
    df_to_export = st.session_state.dataframe_to_export

    if data is not None:
        if st.sidebar.button("Exporter le tableau affiché (CSV)"):
            if df_to_export is not None:
                csv_data = df_to_export.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(
                    label="Télécharger le fichier CSV",
                    data=csv_data,
                    file_name=f"{source_label.replace(' ', '_')}_export.csv",
                    mime="text/csv",
                    key=f"download_csv_{selected_collection_name if selected_collection_name else 'local'}"
                )
            else:
                st.sidebar.warning("Aucun tableau à exporter n'est actuellement affiché.")

        # Add Excel export option
        if st.sidebar.button("Exporter le tableau affiché (Excel)"):
            if df_to_export is not None:
                excel_buffer = io.BytesIO()
                df_to_export.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.sidebar.download_button(
                    label="Télécharger le fichier Excel",
                    data=excel_buffer,
                    file_name=f"{source_label.replace(' ', '_')}_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_excel_{selected_collection_name if selected_collection_name else 'local'}"
                )
            else:
                st.sidebar.warning("Aucun tableau à exporter n'est actuellement affiché.")


        # Add HTML export option
        if st.sidebar.button("Exporter le rapport complet (HTML)"):
            if data is not None:
                num_submissions = data['_index'].nunique() if '_index' in data.columns else 'N/A'

                tables_html = ""
                # Capture HTML of the displayed dataframes
                if 'selected_columns' in locals() and selected_columns:
                    tables_html += "<h3>Tableau des colonnes sélectionnées :</h3>"
                    tables_html += data[selected_columns].to_html(index=False)
                if 'aggregated_data' in locals() and aggregated_data is not None:
                     tables_html += "<h3>Tableau des statistiques agrégées :</h3>"
                     tables_html += aggregated_data.to_html(index=False)

                charts_html = ""
                # Capture HTML of the generated plotly figure
                if 'fig' in locals() and fig is not None:
                     charts_html = fig.to_html(full_html=False)

                html_content = generate_html_report(data, num_submissions, data.columns.tolist(), tables_html, charts_html)

                # Fournir le lien de téléchargement
                st.sidebar.download_button(
                    label="Télécharger le rapport HTML",
                    data=html_content,
                    file_name="rapport_datasolution.html",
                    mime="text/html"
                )
            else:
                st.sidebar.info("Chargez des données pour exporter.")

    else:
        st.sidebar.info("Chargez des données pour exporter.")

    if data is not None:
        # Calculate number of unique submissions using '_index'
        if '_index' in data.columns:
            num_submissions = data['_index'].nunique()
            # Use markdown with HTML/CSS for larger, bordered text
            st.markdown(
                f"""
                <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; font-size: 20px;">
                    Nombre de formulaires soumis : <strong>{num_submissions}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning(f"La colonne '_index' n'a pas été trouvée dans les données de '{source_label}'. Impossible de compter les soumissions.")

        st.write(f"Colonnes présentes dans '{source_label}':")
        st.write(data.columns.tolist()) # Display columns as a list

        # Define categorical and numerical columns
        categorical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
        numerical_columns = data.select_dtypes(include=['number']).columns.tolist()

        # Data Analysis and Visualization
        st.subheader("Analyse et Visualisation des Données")

        # Initialize session state for analyses if it doesn't exist
        if 'analyses' not in st.session_state:
            st.session_state.analyses = []

        # Buttons to add new analyses
        col_add1, col_add2, col_add3 = st.columns(3)
        with col_add1:
            if st.button("Ajouter une Analyse Tableau Agrégé"):
                st.session_state.analyses.append({'type': 'aggregated_table', 'params': {}, 'result': None})
        with col_add2:
            if st.button("Ajouter une Analyse Graphique"):
                st.session_state.analyses.append({'type': 'graph', 'params': {}, 'result': None})
        with col_add3:
            if st.button("Ajouter une Analyse Statistique Descriptive"):
                st.session_state.analyses.append({'type': 'descriptive_stats', 'params': {}, 'result': None})

        # Display and configure analyses
        for i, analysis in enumerate(st.session_state.analyses):
            st.markdown(f"---")
            if analysis['type'] == 'aggregated_table':
                st.subheader(f"Analyse Tableau Agrégé {i+1}")
                # Aggregated table analysis configuration
                group_by_columns = st.multiselect(
                    f"Sélectionnez les variables pour le regroupement (agrégation) pour le Tableau Agrégé {i+1}:",
                    categorical_columns,
                    default=analysis['params'].get('group_by_columns', []),
                    key=f"agg_table_groupby_{i}"
                )

                agg_column = st.selectbox(
                    f"Sélectionnez la variable numérique à agréger pour le Tableau Agrégé {i+1}:",
                    numerical_columns,
                    index=numerical_columns.index(analysis['params'].get('agg_column', numerical_columns[0])) if analysis['params'].get('agg_column') in numerical_columns else 0,
                    key=f"agg_table_agg_col_{i}"
                )

                agg_method = st.selectbox(
                    f"Sélectionnez la méthode d'agrégation pour le Tableau Agrégé {i+1}:",
                    ('count', 'mean', 'sum', 'min', 'max', 'std'),
                    index=('count', 'mean', 'sum', 'min', 'max', 'std').index(analysis['params'].get('agg_method', 'count')),
                    key=f"agg_table_agg_method_{i}"
                )

                if st.button(f"Exécuter l'Analyse Tableau Agrégé {i+1}", key=f"run_agg_table_{i}"):
                    if group_by_columns and agg_column is not None and agg_method is not None:
                        try:
                            aggregated_data = data.groupby(group_by_columns)[agg_column].agg(agg_method).reset_index()
                            st.session_state.analyses[i]['result'] = aggregated_data
                            st.session_state.analyses[i]['params'] = {
                                'group_by_columns': group_by_columns,
                                'agg_column': agg_column,
                                'agg_method': agg_method
                            }
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors de l'agrégation pour le Tableau Agrégé {i+1}: {e}")
                    else:
                        st.warning("Veuillez sélectionner les variables pour le regroupement, la variable numérique et la méthode d'agrégation.")

                # Display aggregated table result
                if analysis['result'] is not None:
                    st.write(f"Résultat de l'Analyse Tableau Agrégé {i+1}:")
                    st.dataframe(analysis['result'])

            elif analysis['type'] == 'graph':
                st.subheader(f"Analyse Graphique {i+1}")
                # Graph analysis configuration
                graph_analysis_type = st.selectbox(
                    f"Choisissez le type de graphique pour l'Analyse Graphique {i+1}:",
                    ('Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram', 'Box Plot', 'Violin Plot', 'Heatmap', '3D Scatter Plot', 'Pair Plot'),
                    index=('Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram', 'Box Plot', 'Violin Plot', 'Heatmap', '3D Scatter Plot', 'Pair Plot').index(analysis['params'].get('chart_type', 'Bar Chart')),
                    key=f"graph_type_{i}"
                )

                # Add aggregation options for graphs
                st.subheader("Statistiques Agrégées pour les Graphiques")
                group_by_columns_graph = st.multiselect(
                    f"Sélectionnez les variables pour le regroupement (agrégation) pour les graphiques pour l'Analyse Graphique {i+1}:",
                    categorical_columns,
                    default=analysis['params'].get('group_by_columns_graph', []),
                    key=f"graph_groupby_{i}"
                )

                agg_column_graph = st.selectbox(
                    f"Sélectionnez la variable numérique à agréger pour les graphiques pour l'Analyse Graphique {i+1}:",
                    numerical_columns,
                    index=numerical_columns.index(analysis['params'].get('agg_column_graph', numerical_columns[0])) if analysis['params'].get('agg_column_graph') in numerical_columns else 0,
                    key=f"graph_agg_col_{i}"
                )

                agg_method_graph = st.selectbox(
                    f"Sélectionnez la méthode d'agrégation pour les graphiques pour l'Analyse Graphique {i+1}:",
                    ('count', 'mean', 'sum', 'min', 'max', 'std'),
                    index=('count', 'mean', 'sum', 'min', 'max', 'std').index(analysis['params'].get('agg_method_graph', 'count')),
                    key=f"graph_agg_method_{i}"
                )

                aggregated_data_graph = None # Initialize aggregated_data_graph
                if group_by_columns_graph and agg_column_graph is not None and agg_method_graph is not None:
                    try:
                        aggregated_data_graph = data.groupby(group_by_columns_graph)[agg_column_graph].agg(agg_method_graph).reset_index()
                    except Exception as e:
                        st.error(f"Erreur lors de l'agrégation pour les graphiques pour l'Analyse Graphique {i+1}: {e}")

                chart_columns = aggregated_data_graph.columns.tolist() if aggregated_data_graph is not None else data.columns.tolist()
                x_column = st.selectbox(
                    f"Sélectionnez la variable pour l'axe X pour l'Analyse Graphique {i+1}:",
                    chart_columns,
                    index=chart_columns.index(analysis['params'].get('x_column', chart_columns[0])) if analysis['params'].get('x_column') in chart_columns else 0,
                    key=f"graph_x_{i}"
                )

                y_column = st.selectbox(
                    f"Sélectionnez la variable pour l'axe Y pour l'Analyse Graphique {i+1}:",
                    chart_columns,
                    index=chart_columns.index(analysis['params'].get('y_column', chart_columns[0])) if analysis['params'].get('y_column') in chart_columns else 0,
                    key=f"graph_y_{i}"
                )

                color_column = st.selectbox(
                    f"Sélectionnez la variable pour la couleur (optionnel) pour l'Analyse Graphique {i+1}:",
                    ['None'] + categorical_columns,
                    index=0,
                    key=f"graph_color_{i}"
                )

                size_column = st.selectbox(
                    f"Sélectionnez la variable pour la taille (optionnel) pour l'Analyse Graphique {i+1}:",
                    ['None'] + numerical_columns,
                    index=0,
                    key=f"graph_size_{i}"
                )

                if st.button(f"Exécuter l'Analyse Graphique {i+1}", key=f"run_graph_{i}"):
                     if x_column and y_column:
                        try:
                            # Use Plotly for charting
                            if graph_analysis_type == 'Bar Chart':
                                fig = px.bar(aggregated_data_graph if aggregated_data_graph is not None else data, x=x_column, y=y_column, color=color_column if color_column != 'None' else None, title=f'{y_column} by {x_column}')
                            elif graph_analysis_type == 'Line Chart':
                                fig = px.line(aggregated_data_graph if aggregated_data_graph is not None else data, x=x_column, y=y_column, color=color_column if color_column != 'None' else None, title=f'{y_column} over {x_column}')
                            elif graph_analysis_type == 'Scatter Plot':
                                fig = px.scatter(aggregated_data_graph if aggregated_data_graph is not None else data, x=x_column, y=y_column, color=color_column if color_column != 'None' else None, size=size_column if size_column != 'None' else None, title=f'{y_column} vs {x_column}')
                            elif graph_analysis_type == 'Histogram':
                                 if x_column in numerical_columns:
                                     fig = px.histogram(aggregated_data_graph if aggregated_data_graph is not None else data, x=x_column, color=color_column if color_column != 'None' else None, title=f'Histogram of {x_column}')
                                 else:
                                     st.warning(f"Histogram requires a numerical column for the X-axis. '{x_column}' is not numerical.")
                                     fig = None
                            elif graph_analysis_type == 'Box Plot':
                                fig = px.box(aggregated_data_graph if aggregated_data_graph is not None else data, x=x_column, y=y_column, color=color_column if color_column != 'None' else None, title=f'Box Plot of {y_column} by {x_column}')
                            elif graph_analysis_type == 'Violin Plot':
                                fig = px.violin(aggregated_data_graph if aggregated_data_graph is not None else data, x=x_column, y=y_column, color=color_column if color_column != 'None' else None, title=f'Violin Plot of {y_column} by {x_column}')
                            elif graph_analysis_type == 'Heatmap':
                                fig = px.density_heatmap(aggregated_data_graph if aggregated_data_graph is not None else data, x=x_column, y=y_column, title=f'Heatmap of {y_column} by {x_column}')
                            elif graph_analysis_type == '3D Scatter Plot':
                                fig = px.scatter_3d(aggregated_data_graph if aggregated_data_graph is not None else data, x=x_column, y=y_column, z=size_column if size_column != 'None' else None, color=color_column if color_column != 'None' else None, title=f'3D Scatter Plot of {y_column} vs {x_column}')
                            elif graph_analysis_type == 'Pair Plot':
                                fig = px.scatter_matrix(aggregated_data_graph if aggregated_data_graph is not None else data, dimensions=[x_column, y_column, size_column if size_column != 'None' else None, color_column if color_column != 'None' else None], title=f'Pair Plot')

                            if fig is not None:
                                fig.update_layout(legend=dict(title=color_column if color_column != 'None' else ''))
                                fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
                                st.session_state.analyses[i]['result'] = fig
                                st.session_state.analyses[i]['params'] = {
                                    'chart_type': graph_analysis_type,
                                    'group_by_columns_graph': group_by_columns_graph,
                                    'agg_column_graph': agg_column_graph,
                                    'agg_method_graph': agg_method_graph,
                                    'x_column': x_column,
                                    'y_column': y_column,
                                    'color_column': color_column,
                                    'size_column': size_column
                                }
                                st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors de la génération du graphique pour l'Analyse Graphique {i+1}: {e}")
                     else:
                        st.warning("Veuillez sélectionner les variables pour les axes X et Y.")

                # Display graph result
                if analysis['result'] is not None:
                    st.write(f"Résultat de l'Analyse Graphique {i+1}:")
                    st.plotly_chart(analysis['result'], use_container_width=True)

            elif analysis['type'] == 'descriptive_stats':
                st.subheader(f"Analyse Statistique Descriptive {i+1}")
                # Descriptive stats analysis configuration
                selected_columns = st.multiselect(
                    f"Sélectionnez les colonnes pour l'Analyse Statistique Descriptive {i+1}:",
                    numerical_columns,
                    default=analysis['params'].get('selected_columns', []),
                    key=f"desc_stats_columns_{i}"
                )

                if st.button(f"Exécuter l'Analyse Statistique Descriptive {i+1}", key=f"run_desc_stats_{i}"):
                    if selected_columns:
                        try:
                            descriptive_stats = data[selected_columns].describe()
                            st.session_state.analyses[i]['result'] = descriptive_stats
                            st.session_state.analyses[i]['params'] = {
                                'selected_columns': selected_columns
                            }
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors du calcul des statistiques descriptives pour l'Analyse Statistique Descriptive {i+1}: {e}")
                    else:
                        st.warning("Veuillez sélectionner au moins une colonne pour l'analyse.")

                # Display descriptive stats result
                if analysis['result'] is not None:
                    st.write(f"Résultat de l'Analyse Statistique Descriptive {i+1}:")
                    st.dataframe(analysis['result'])

            elif analysis['type'] == 'advanced_stats':
                st.subheader(f"Analyse Statistique Avancée {i+1}")
                # Advanced stats analysis configuration
                advanced_analysis_type = st.selectbox(
                    f"Sélectionnez le type d'analyse avancée pour l'Analyse Statistique Avancée {i+1}:",
                    ('Test T', 'ANOVA', 'Chi-Square Test', 'Corrélation', 'Régression Linéaire', 'ACP', 'Clustering', 'Détection d\'Anomalies'),
                    index=('Test T', 'ANOVA', 'Chi-Square Test', 'Corrélation', 'Régression Linéaire', 'ACP', 'Clustering', 'Détection d\'Anomalies').index(analysis['params'].get('advanced_analysis_type', 'Test T')),
                    key=f"advanced_stats_type_{i}"
                )

                # The configuration and execution logic for each advanced analysis type
                # will need to be added here, similar to the table and graph analyses.
                # For now, I will keep the existing advanced stats section as a placeholder
                # and integrate it into the loop structure later if needed.
                st.info("Configuration and execution for advanced statistical analyses will be added here.")

        # Add button to show/hide advanced analysis section
        if 'show_advanced_analysis' not in st.session_state:
            st.session_state.show_advanced_analysis = False

        if st.button("Afficher/Masquer les Analyses Statistiques Avancées"):
            st.session_state.show_advanced_analysis = not st.session_state.show_advanced_analysis

        if st.session_state.show_advanced_analysis:
            st.subheader("Analyses Statistiques Avancées")

            advanced_analysis_type = st.selectbox(
                "Sélectionnez le type d'analyse avancée :",
                ('Test T', 'ANOVA', 'Chi-Square Test', 'Corrélation', 'Régression Linéaire', 'ACP', 'Clustering', 'Détection d\'Anomalies')
            )

            if advanced_analysis_type == 'Test T':
                st.write("Test T pour comparer les moyennes de deux groupes.")
                group1 = st.selectbox("Sélectionnez la première variable catégorielle :", categorical_columns)
                group2 = st.selectbox("Sélectionnez la deuxième variable catégorielle :", categorical_columns)
                numeric_var = st.selectbox("Sélectionnez la variable numérique :", numerical_columns)

                if st.button("Effectuer le Test T"):
                    group1_data = data[data[group1] == data[group1].unique()[0]][numeric_var]
                    group2_data = data[data[group1] == data[group1].unique()[1]][numeric_var]
                    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                    st.write(f"T-Statistic: {t_stat}, P-Value: {p_value}")

            elif advanced_analysis_type == 'ANOVA':
                st.write("ANOVA pour comparer les moyennes de plusieurs groupes.")
                anova_groups = st.multiselect("Sélectionnez les variables catégorielles :", categorical_columns)
                anova_numeric_var = st.selectbox("Sélectionnez la variable numérique :", numerical_columns)

                if st.button("Effectuer l'ANOVA"):
                    grouped_data = [data[data[group] == value][anova_numeric_var] for group in anova_groups for value in data[group].unique()]
                    f_stat, p_value = stats.f_oneway(*grouped_data)
                    st.write(f"F-Statistic: {f_stat}, P-Value: {p_value}")

            elif advanced_analysis_type == 'Chi-Square Test':
                st.write("Test du Chi-carré pour vérifier l'indépendance entre deux variables catégorielles.")
                chi2_var1 = st.selectbox("Sélectionnez la première variable catégorielle :", categorical_columns)
                chi2_var2 = st.selectbox("Sélectionnez la deuxième variable catégorielle :", categorical_columns)

                if st.button("Effectuer le Test du Chi-carré"):
                    contingency_table = pd.crosstab(data[chi2_var1], data[chi2_var2])
                    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    st.write(f"Chi2-Statistic: {chi2_stat}, P-Value: {p_value}")

            elif advanced_analysis_type == 'Corrélation':
                st.write("Analyse de corrélation entre deux variables numériques.")
                corr_var1 = st.selectbox("Sélectionnez la première variable numérique :", numerical_columns)
                corr_var2 = st.selectbox("Sélectionnez la deuxième variable numérique :", numerical_columns)

                if st.button("Calculer la Corrélation"):
                    correlation = data[[corr_var1, corr_var2]].corr().iloc[0, 1]
                    st.write(f"Corrélation entre {corr_var1} et {corr_var2} : {correlation}")

            elif advanced_analysis_type == 'Régression Linéaire':
                st.write("Régression linéaire pour prédire une variable numérique.")
                reg_target = st.selectbox("Sélectionnez la variable cible :", numerical_columns)
                reg_features = st.multiselect("Sélectionnez les variables explicatives :", numerical_columns)

                if st.button("Effectuer la Régression Linéaire"):
                    X = data[reg_features]
                    y = data[reg_target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"MSE: {mse}, R²: {r2}")

            elif advanced_analysis_type == 'ACP':
                st.write("Analyse en Composantes Principales (ACP).")
                pca_features = st.multiselect("Sélectionnez les variables numériques :", numerical_columns)

                if st.button("Effectuer l'ACP"):
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(data[pca_features])
                    st.write("Résultats de l'ACP :")
                    st.dataframe(pd.DataFrame(data=pca_result, columns=['PC1', 'PC2']))
                    st.write("Variance expliquée par composante :", pca.explained_variance_ratio_)

            elif advanced_analysis_type == 'Clustering':
                st.write("Clustering pour regrouper les données.")
                cluster_features = st.multiselect("Sélectionnez les variables numériques :", numerical_columns)
                num_clusters = st.number_input("Nombre de clusters :", min_value=2, max_value=10, value=3)

                if st.button("Effectuer le Clustering"):
                    kmeans = KMeans(n_clusters=num_clusters)
                    data['Cluster'] = kmeans.fit_predict(data[cluster_features])
                    st.write("Résultats du clustering :")
                    st.dataframe(data)

            elif advanced_analysis_type == 'Détection d\'Anomalies':
                st.write("Détection d'anomalies dans les données.")
                anomaly_features = st.multiselect("Sélectionnez les variables numériques :", numerical_columns)

                if st.button("Détecter les Anomalies"):
                    # Simple anomaly detection using Z-score
                    z_scores = np.abs(stats.zscore(data[anomaly_features]))
                    anomalies = np.where(z_scores > 3, True, False)
                    data['Anomaly'] = anomalies.any(axis=1)
                    st.write(f"Résultats de la détection d'anomalies :")
                    st.dataframe(data)


with manual_tab:
    st.markdown(f"## 📘 Manuel d'Utilisation — {APP_NAME}")

    st.markdown(f"""
Bienvenue dans **{APP_NAME}**.
Cette application facilite l'exploitation et l'analyse de vos données collectées.

---

### 🔹 1️⃣ Sélection des Données

- **URL API KoboToolbox** : charge les données depuis une URL d'export Kobo (.xlsx ou .csv), prédéfinie ou personnalisée.
- **Fichier Excel** : importez un fichier `.xlsx` — les en-têtes sont détectés automatiquement.
- **Fichier CSV** : importez un fichier `.csv` — la première ligne devient les en-têtes.
- **Texte simple** : collez des données tabulaires (CSV, point-virgule ou tabulation) — l'app crée le DataFrame avec les en-têtes.

- **Actualisation** : bouton « Actualiser les données » dans la barre latérale.

---

### 🔹 2️⃣ Analyse et Visualisation

- Tableaux agrégés, graphiques Plotly, statistiques descriptives et analyses avancées (Tests T, ANOVA, ACP, clustering…).

---

### 🔹 3️⃣ Exportation

- CSV, Excel et rapport HTML.

---

### 🔹 4️⃣ Personnalisation

- Renommage des colonnes dans la barre latérale.

---

**👨‍💻 DataSolution — Sidoine YEBADOKPO**
Programmeur & Data Scientist

📂 [Portfolio](https://huggingface.co/spaces/Sidoineko/portfolio)
📞 +229 01 96 91 13 46
🔗 [LinkedIn](https://www.linkedin.com/in/sidoineko)
""")

with chat_tab:
    st.markdown("## Chat avec Gemini")
    st.write("Posez des questions sur les données ou les analyses effectuées.")

    if not google_api_key:
        st.warning("Clé API Google Gemini non configurée. Ajoutez GOOGLE_API_KEY dans .env ou les Secrets Streamlit.")

    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add a text input field for the user to type their questions
    user_question = st.chat_input("Votre question :")

    if user_question and google_api_key:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        try:
            # Create a Gemini model instance
            model = genai.GenerativeModel('gemini-1.5-flash-latest')

            # Send the message to the model
            response = model.generate_content(user_question)

            # Add Gemini response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response.text})

            st.rerun() # Rerun to update the chat display

        except Exception as e:
            st.error(f"Erreur lors de la communication avec Gemini : {e}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Erreur: {e}"})
            st.rerun()
