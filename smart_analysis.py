# -*- coding: utf-8 -*-
"""
smart_analysis.py
==================
Moteur d'analyse intelligente pour CCR-B DataViz.

Ce module remplace les anciennes fonctions d'analyse "génériques" par un
moteur qui :
  1. Profile automatiquement n'importe quel dataset chargé (Kobo ou autre) :
     détecte le type réel de chaque variable (numérique, catégorielle,
     date, identifiant, géographique, échelle de Likert/satisfaction...).
  2. Calcule des statistiques pertinentes (concentration, asymétrie,
     valeurs aberrantes, corrélations, associations catégorielles,
     tendances temporelles...) et les traduit en phrases lisibles.
  3. Choisit automatiquement le type de graphique Plotly le plus parlant
     pour chaque variable / relation, avec un thème visuel cohérent.
  4. Fournit un rendu Streamlit interactif ("render_smart_dashboard")
     et une liste d'insights structurés réutilisable pour le rapport
     Word (voir report_builder.py).

Aucune dépendance à l'ancien code n'est requise : ce module est autonome,
il suffit de lui passer un DataFrame pandas.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:  # pragma: no cover
    _HAS_STREAMLIT = False


# =============================================================================
# THÈME VISUEL
# =============================================================================
PRIMARY = "#1B5E20"
PALETTE = [
    "#1B5E20", "#43A047", "#8BC34A", "#FDD835", "#FB8C00",
    "#EF5350", "#42A5F5", "#5C6BC0", "#8D6E63", "#26A69A",
]
SEQ_GREEN = ["#F1F8E9", "#C5E1A5", "#8BC34A", "#558B2F", "#1B5E20"]
DIVERGING = ["#C62828", "#EF9A9A", "#F5F5F5", "#A5D6A7", "#1B5E20"]
FONT_FAMILY = "Segoe UI, Helvetica, Arial, sans-serif"


def style_fig(fig: go.Figure, height: int = 420, show_legend: Optional[bool] = None) -> go.Figure:
    """Applique un thème visuel cohérent et soigné à une figure Plotly."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=13, color="#263238"),
        title=dict(font=dict(size=17, color="#1B5E20", family=FONT_FAMILY)),
        margin=dict(l=40, r=30, t=60, b=40),
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT_FAMILY),
        colorway=PALETTE,
    )
    if show_legend is not None:
        fig.update_layout(showlegend=show_legend)
    fig.update_xaxes(showgrid=False, linecolor="#CFD8DC")
    fig.update_yaxes(showgrid=True, gridcolor="#ECEFF1", zeroline=False)
    return fig


# =============================================================================
# DÉTECTION DES TYPES DE VARIABLES
# =============================================================================
ID_KEYWORDS = ["id", "uuid", "_uuid", "key", "instanceid", "submission_id", "index", "_index"]
PHONE_KEYWORDS = ["telephone", "téléphone", "tel_", "_tel", "phone", "numero_tel", "n°_tel", "num_tel", "contact_"]
METADATA_KEYWORDS = ["enqueteur", "enquêteur", "agent_enquêteur", "agent_enqueteur", "nom_de_lenqueteur",
                     "nom_et_prenoms_de_lagent"]
DATE_KEYWORDS = ["date", "_date", "submission", "jour", "created", "soumission", "start", "end", "today"]
GEO_LAT_KEYWORDS = ["latitude", "_lat", "lat_"]
GEO_LON_KEYWORDS = ["longitude", "_lon", "_lng", "lon_"]
LOCATION_KEYWORDS = ["commune", "ville", "village", "region", "région", "département", "departement",
                     "arrondissement", "zone", "localite", "localité", "quartier", "district",
                     "pda", "pôle", "pole", "province", "circonscription"]
LIKERT_KEYWORDS = ["satisf", "accord", "note", "avis", "echelle", "échelle", "qualite", "qualité",
                   "score", "evaluation", "évaluation", "niveau_"]
BOOL_TRUE_SET = {"1", "oui", "yes", "true", "vrai", "o"}
BOOL_FALSE_SET = {"0", "non", "no", "false", "faux", "n"}

# --- Rôles sémantiques métier : permet à l'outil de "comprendre" l'objet de la collecte
# (démographie, production agricole, structuration organisationnelle, accès aux facteurs de
# production...) et de calculer des indicateurs directement exploitables, plutôt que de se
# limiter à des statistiques génériques sans lecture métier.
SEMANTIC_KEYWORDS = {
    "gender": ["sexe", "genre"],
    "age": ["âge", "age"],
    "education": ["instruction", "scolar", "diplôme"],
    "surface": ["superficie", "hectare", "emblav"],
    "production": ["récolt", "recolt", "rendement", "quantité_tonne", "quantite_tonne", "production"],
    "experience": ["expérience", "experience", "ancienneté", "années_d", "annees_d"],
    "cooperative_membership": ["membre_dune_coopérative", "membre_dune_cooperative", "membre_coopérative",
                               "membre_association", "membre_dun_groupe"],
    "cooperative_size": ["effectif", "taille_de_la_coopérative", "taille_du_groupe"],
    "cluster_membership": ["cluster", "appartenez_vous"],
    "water_access": ["disposez_vous_de_leau", "accès_à_leau", "acces_a_leau", "irrigation"],
    "site_amenagement": ["site_est_il_aménagé", "site_est_il_amenage", "aménagé"],
    "phone_ownership": ["téléphone", "telephone"],
}

THEME_KEYWORDS = {
    "agriculture": ["riz", "paddy", "semence", "récolt", "recolt", "superficie", "hectare",
                    "rendement", "culture", "exploitation", "agricole", "riziculture"],
    "sante": ["santé", "sante", "maladie", "vaccin", "clinique", "hôpital", "hopital", "medical", "médical"],
    "education": ["scolar", "école", "ecole", "élève", "eleve", "enseignant", "classe"],
    "satisfaction": ["satisf", "accord", "avis", "évaluation", "evaluation", "qualité_du_service"],
    "organisation": ["coopérative", "cooperative", "association", "groupement", "cluster"],
    "eau_assainissement": ["eau_potable", "assainissement", "latrine", "hygiène", "hygiene"],
    "economie": ["revenu", "salaire", "budget", "épargne", "epargne", "crédit", "credit"],
}


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def _cols_matching(df_columns, keywords) -> list:
    """Associe des colonnes à des mots-clés métier. Pour éviter les faux positifs du type
    'age' détecté dans 'agent' ou 'village', les mots-clés courts (<=4 caractères) doivent
    correspondre à un token entier délimité par '_' ; les mots-clés plus longs (peu ambigus)
    peuvent matcher en sous-chaîne."""
    result = []
    for c in df_columns:
        text = _norm(c)
        tokens = [t for t in re.split(r"[^a-zà-ÿ0-9]+", text) if t]
        matched = False
        for k in keywords:
            k_norm = _norm(k)
            if len(k_norm) <= 4:
                if k_norm in tokens:
                    matched = True
                    break
            else:
                if k_norm in text:
                    matched = True
                    break
        if matched:
            result.append(c)
    return result


def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Retourne un dict {colonne: type_detecte} avec type_detecte parmi :
    'id', 'datetime', 'boolean', 'geo_lat', 'geo_lon', 'likert',
    'categorical_numeric', 'numeric', 'constant', 'text_free',
    'categorical', 'geo_categorical'
    """
    n_rows = len(df)
    types = {}
    for col in df.columns:
        s = df[col]
        col_l = _norm(col)
        nonnull = s.dropna()
        nun = s.nunique(dropna=True)
        pct_unique = (nun / n_rows * 100) if n_rows else 0

        # --- Identifiants ---
        if any(k in col_l for k in ID_KEYWORDS) and pct_unique > 85:
            types[col] = "id"
            continue
        if any(k in col_l for k in PHONE_KEYWORDS):
            types[col] = "id"
            continue

        # --- Constante ---
        if nun <= 1:
            types[col] = "constant"
            continue

        # --- Dates ---
        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
            continue
        is_stringish = (s.dtype == object) or pd.api.types.is_string_dtype(s)
        if is_stringish and any(k in col_l for k in DATE_KEYWORDS):
            sample = nonnull.astype(str).head(40)
            try:
                parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
                if parsed.notna().mean() > 0.7:
                    types[col] = "datetime"
                    continue
            except Exception:
                pass

        # --- Géographie (coordonnées) ---
        if pd.api.types.is_numeric_dtype(s):
            if any(k in col_l for k in GEO_LAT_KEYWORDS) and nonnull.between(-90, 90).mean() > 0.9:
                types[col] = "geo_lat"
                continue
            if any(k in col_l for k in GEO_LON_KEYWORDS) and nonnull.between(-180, 180).mean() > 0.9:
                types[col] = "geo_lon"
                continue

        # --- Booléens ---
        if nun == 2:
            vals = set(str(v).strip().lower() for v in nonnull.unique())
            if vals <= (BOOL_TRUE_SET | BOOL_FALSE_SET):
                types[col] = "boolean"
                continue
            if pd.api.types.is_bool_dtype(s):
                types[col] = "boolean"
                continue

        # --- Numérique ---
        if pd.api.types.is_numeric_dtype(s):
            is_int_like = nonnull.apply(lambda x: float(x).is_integer()).mean() > 0.98 if len(nonnull) else False
            if nun <= 8 and is_int_like and any(k in col_l for k in LIKERT_KEYWORDS):
                types[col] = "likert"
            elif nun <= 12 and is_int_like and pct_unique < 30:
                types[col] = "categorical_numeric"
            else:
                types[col] = "numeric"
            continue

        # --- Texte / catégoriel ---
        avg_len = nonnull.astype(str).str.len().mean() if len(nonnull) else 0
        if any(k in col_l for k in LOCATION_KEYWORDS) and nun <= 60:
            types[col] = "geo_categorical"
        elif pct_unique > 55 or (avg_len and avg_len > 45):
            types[col] = "text_free"
        elif nun <= 40:
            types[col] = "categorical"
        else:
            types[col] = "text_free"

    return types


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    types: dict
    numeric_cols: list
    categorical_cols: list
    likert_cols: list
    boolean_cols: list
    datetime_cols: list
    geo_categorical_cols: list
    geo_lat: Optional[str]
    geo_lon: Optional[str]
    id_cols: list
    constant_cols: list
    text_free_cols: list
    missing_pct: pd.Series
    duplicate_rows: int
    completeness: float


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """Construit le profil global du dataset."""
    types = detect_column_types(df)

    def cols_of(t):
        return [c for c, v in types.items() if v == t]

    geo_lat = next(iter(cols_of("geo_lat")), None)
    geo_lon = next(iter(cols_of("geo_lon")), None)

    missing_pct = (df.isna().mean() * 100).round(1)
    completeness = round(100 - missing_pct.mean(), 1) if len(df.columns) else 100.0

    return DataProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        types=types,
        numeric_cols=cols_of("numeric"),
        categorical_cols=cols_of("categorical") + cols_of("categorical_numeric") + cols_of("boolean"),
        likert_cols=cols_of("likert"),
        boolean_cols=cols_of("boolean"),
        datetime_cols=cols_of("datetime"),
        geo_categorical_cols=cols_of("geo_categorical"),
        geo_lat=geo_lat,
        geo_lon=geo_lon,
        id_cols=cols_of("id"),
        constant_cols=cols_of("constant"),
        text_free_cols=cols_of("text_free"),
        missing_pct=missing_pct,
        duplicate_rows=int(df.duplicated().sum()),
        completeness=completeness,
    )


# =============================================================================
# COMPRÉHENSION DE L'OBJECTIF DE LA COLLECTE
# =============================================================================
@dataclass
class SurveyContext:
    themes: list                 # thèmes métier détectés, ex: ["agriculture", "organisation"]
    theme_label: str             # phrase lisible décrivant l'enquête
    roles: dict                  # rôle sémantique -> liste de colonnes correspondantes
    zone_col: Optional[str]      # meilleure colonne de comparaison territoriale
    zone_candidates: list        # autres colonnes géographiques possibles
    yield_pairs: list            # [(colonne_production, colonne_surface, label_periode)]


def detect_semantic_roles(df: pd.DataFrame, profile: "DataProfile") -> dict:
    """Associe à chaque rôle métier (âge, sexe, superficie, production...) les colonnes
    du dataset qui y correspondent, en croisant mots-clés et type détecté."""
    roles = {}
    all_cols = list(df.columns)
    for role, keywords in SEMANTIC_KEYWORDS.items():
        matches = _cols_matching(all_cols, keywords)
        # ne garder que les colonnes dont le type détecté est cohérent avec un usage analytique
        matches = [c for c in matches if profile.types.get(c) not in ("id", "constant", "text_free")]
        if matches:
            roles[role] = matches
    return roles


def infer_survey_theme(df: pd.DataFrame) -> tuple:
    """Détecte le(s) thème(s) métier probable(s) de la collecte à partir des noms de colonnes."""
    all_text = " ".join(_norm(c) for c in df.columns)
    scores = {}
    for theme, keywords in THEME_KEYWORDS.items():
        score = sum(all_text.count(k) for k in keywords)
        if score > 0:
            scores[theme] = score
    themes = sorted(scores, key=lambda t: -scores[t])[:3]

    labels = {
        "agriculture": "une enquête de caractérisation de producteurs / exploitations agricoles",
        "sante": "une enquête à composante santé",
        "education": "une enquête à composante éducation",
        "satisfaction": "une enquête de satisfaction / évaluation de service",
        "organisation": "une enquête portant notamment sur la structuration en organisations (coopératives, groupements)",
        "eau_assainissement": "une enquête à composante eau / assainissement",
        "economie": "une enquête à composante socio-économique",
    }
    if themes:
        theme_label = "Ce jeu de données correspond vraisemblablement à " + " et ".join(
            labels.get(t, t) for t in themes[:2]
        ) + "."
    else:
        theme_label = "Le thème précis de cette collecte n'a pas pu être déterminé automatiquement à partir des noms de variables."
    return themes, theme_label


def find_zone_column(df: pd.DataFrame, profile: "DataProfile") -> tuple:
    """Choisit la meilleure colonne pour des comparaisons territoriales (ni trop fine, ni trop agrégée)."""
    candidates = list(profile.geo_categorical_cols)
    if not candidates:
        return None, candidates
    # Prioriser un nombre de zones "lisible" pour un graphique comparatif (entre 2 et 9 modalités)
    def _score(c):
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= 9:
            return (0, nun)   # meilleur score : peu de modalités, lisible
        return (1, nun)       # sinon, toujours utilisable mais moins prioritaire
    ranked = sorted(candidates, key=_score)
    return ranked[0], candidates


def find_yield_pairs(df: pd.DataFrame, roles: dict) -> list:
    """Tente d'apparier automatiquement une colonne de production avec la colonne de superficie
    correspondante (même période), pour calculer un rendement (kg/ha) — un indicateur clé et
    directement exploitable pour toute enquête agricole, mais absent des données brutes."""
    prod_cols = roles.get("production", [])
    surf_cols = roles.get("surface", [])
    pairs = []
    used_surf = set()
    time_markers = ["cette_année", "cette_annee", "en_cours", "actuel", "passée", "passee",
                    "lannée_passée", "lannee_passee", "précédente", "precedente", "dernière", "derniere"]
    for p in prod_cols:
        p_l = _norm(p)
        p_marker = next((m for m in time_markers if m in p_l), None)
        best_s = None
        if p_marker:
            for s in surf_cols:
                if s in used_surf:
                    continue
                if p_marker in _norm(s):
                    best_s = s
                    break
        if best_s:
            pairs.append((p, best_s, p_marker))
            used_surf.add(best_s)
    if not pairs and len(prod_cols) == 1 and len(surf_cols) == 1:
        pairs.append((prod_cols[0], surf_cols[0], None))
    return pairs


def build_survey_context(df: pd.DataFrame, profile: "DataProfile") -> SurveyContext:
    roles = detect_semantic_roles(df, profile)
    themes, theme_label = infer_survey_theme(df)
    zone_col, zone_candidates = find_zone_column(df, profile)
    yield_pairs = find_yield_pairs(df, roles)
    return SurveyContext(
        themes=themes, theme_label=theme_label, roles=roles,
        zone_col=zone_col, zone_candidates=zone_candidates, yield_pairs=yield_pairs,
    )


def cramers_v(confusion_matrix: pd.DataFrame, min_n: int = 30) -> tuple:
    """Coefficient de Cramér's V + p-value (test du Chi²). Retourne (v, p_value, n)."""
    try:
        n = int(confusion_matrix.sum().sum())
        if n < min_n:
            return 0.0, 1.0, n
        chi2, p, _, _ = stats.chi2_contingency(confusion_matrix)
        if n == 0:
            return 0.0, 1.0, n
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        if denom <= 0:
            return 0.0, 1.0, n
        v = float(np.sqrt(phi2corr / denom))
        return v, float(p), n
    except Exception:
        return 0.0, 1.0, 0


def correlation_ratio(categories: pd.Series, values: pd.Series, min_n: int = 30, min_group_n: int = 5) -> tuple:
    """Eta (rapport de corrélation) + p-value (ANOVA) entre catégorielle et numérique. Retourne (eta, p_value, n)."""
    try:
        df_tmp = pd.DataFrame({"cat": categories, "val": values}).dropna()
        n = len(df_tmp)
        if n < min_n or df_tmp["cat"].nunique() < 2:
            return 0.0, 1.0, n
        groups = [g["val"].values for _, g in df_tmp.groupby("cat") if len(g) >= min_group_n]
        if len(groups) < 2:
            return 0.0, 1.0, n
        grand_mean = df_tmp["val"].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((df_tmp["val"] - grand_mean) ** 2).sum()
        eta = float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0
        try:
            _, p = stats.f_oneway(*groups)
        except Exception:
            p = 1.0
        return eta, float(p), n
    except Exception:
        return 0.0, 1.0, 0


def concentration_share(value_counts: pd.Series) -> float:
    """Part (%) prise par la modalité la plus fréquente."""
    total = value_counts.sum()
    return float(value_counts.iloc[0] / total * 100) if total else 0.0


def skew_comment(skew: float) -> str:
    if skew > 1:
        return "fortement concentrée vers les faibles valeurs, avec quelques valeurs élevées qui tirent la moyenne vers le haut"
    if skew > 0.5:
        return "légèrement concentrée vers les faibles valeurs"
    if skew < -1:
        return "fortement concentrée vers les valeurs élevées, avec quelques faibles valeurs qui tirent la moyenne vers le bas"
    if skew < -0.5:
        return "légèrement concentrée vers les valeurs élevées"
    return "globalement symétrique autour de la moyenne"


# =============================================================================
# STRUCTURE D'UN INSIGHT
# =============================================================================
@dataclass
class Insight:
    category: str          # ex: "Qualité des données", "Répartitions", "Relations", "Tendances", "Géographie"
    title: str
    narrative: str
    importance: float
    fig: Optional[go.Figure] = None
    table: Optional[pd.DataFrame] = None
    icon: str = "📊"


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%".replace(".0%", "%")


# =============================================================================
# GÉNÉRATION DES INSIGHTS
# =============================================================================
def _insight_data_quality(df: pd.DataFrame, profile: DataProfile) -> list[Insight]:
    insights = []

    completeness = profile.completeness
    narrative = (
        f"Le jeu de données comprend **{profile.n_rows:,} enregistrements** et "
        f"**{profile.n_cols} variables**, pour un taux de complétude global de "
        f"**{_fmt_pct(completeness)}**."
    ).replace(",", " ")
    if profile.duplicate_rows > 0:
        narrative += f" **{profile.duplicate_rows}** ligne(s) apparaissent comme des doublons exacts."
    worst = profile.missing_pct.sort_values(ascending=False)
    worst = worst[worst > 20]
    if len(worst) > 0:
        top_worst = ", ".join(f"`{c}` ({_fmt_pct(v)})" for c, v in worst.head(5).items())
        narrative += f" Les variables les plus incomplètes sont : {top_worst}."
    else:
        narrative += " Aucune variable ne présente de taux de valeurs manquantes préoccupant (>20%)."

    fig = None
    if profile.missing_pct.max() > 0:
        top_missing = profile.missing_pct.sort_values(ascending=False).head(15)
        top_missing = top_missing[top_missing > 0]
        if len(top_missing) > 0:
            fig = px.bar(
                x=top_missing.values, y=top_missing.index, orientation="h",
                labels={"x": "% de valeurs manquantes", "y": ""},
                title="Complétude des données — variables les plus lacunaires",
                color=top_missing.values, color_continuous_scale="Reds",
            )
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            fig = style_fig(fig, height=max(320, 28 * len(top_missing)))

    insights.append(Insight(
        category="Qualité des données",
        title="Vue d'ensemble et complétude",
        narrative=narrative,
        importance=100,
        fig=fig,
        icon="✅",
    ))
    return insights


def _insight_categorical(df: pd.DataFrame, col: str, kind: str) -> Optional[Insight]:
    s = df[col].dropna()
    if len(s) == 0:
        return None
    vc = s.astype(str).value_counts()
    n_modalities = len(vc)
    missing_pct = df[col].isna().mean() * 100
    top_share = concentration_share(vc)
    top_label = vc.index[0]

    # Limiter l'affichage aux 12 modalités les plus fréquentes
    vc_display = vc.head(12).sort_values(ascending=True)
    other_count = vc.iloc[12:].sum() if len(vc) > 12 else 0

    narrative = (
        f"La variable `{col}` compte **{n_modalities} modalités** distinctes "
        f"(complétude : {_fmt_pct(100 - missing_pct)}). "
        f"La modalité la plus fréquente est **« {top_label} »**, représentant "
        f"**{_fmt_pct(top_share)}** des réponses valides."
    )
    if top_share > 70:
        narrative += " La répartition est **très concentrée** sur cette modalité, ce qui peut limiter la diversité des profils observés."
    elif top_share < 25 and n_modalities > 4:
        narrative += " La répartition est **assez équilibrée** entre les différentes modalités."

    fig = px.bar(
        x=vc_display.values, y=vc_display.index, orientation="h",
        labels={"x": "Nombre de réponses", "y": ""},
        title=f"Répartition — {col}",
        text=vc_display.values,
        color=vc_display.values, color_continuous_scale=SEQ_GREEN,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False)
    fig = style_fig(fig, height=max(320, 32 * len(vc_display)))

    importance = 40 + min(30, n_modalities) - (missing_pct * 0.2)
    return Insight(
        category="Répartitions clés",
        title=f"Répartition de « {col} »",
        narrative=narrative,
        importance=importance,
        fig=fig,
        table=vc.rename("Effectif").to_frame(),
        icon="📊",
    )


def _insight_numeric(df: pd.DataFrame, col: str) -> Optional[Insight]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 5:
        return None
    desc = s.describe()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < lo) | (s > hi)]
    skew = s.skew()

    narrative = (
        f"La variable `{col}` a une moyenne de **{desc['mean']:.2f}** "
        f"(médiane : {desc['50%']:.2f}, écart-type : {desc['std']:.2f}), "
        f"avec des valeurs comprises entre {desc['min']:.2f} et {desc['max']:.2f}. "
        f"Sa distribution est {skew_comment(skew)}."
    )
    if len(outliers) > 0:
        pct_out = len(outliers) / len(s) * 100
        narrative += f" **{len(outliers)} valeur(s) atypique(s)** ({_fmt_pct(pct_out)}) sortent de l'intervalle habituel et méritent une vérification."

    fig = px.histogram(
        s.to_frame(name=col), x=col, nbins=min(30, max(10, s.nunique())),
        title=f"Distribution — {col}",
        color_discrete_sequence=[PRIMARY],
    )
    fig.add_vline(x=desc["mean"], line_dash="dash", line_color="#EF5350",
                   annotation_text="Moyenne", annotation_position="top")
    fig = style_fig(fig, height=380)

    importance = 45 + (10 if len(outliers) > 0 else 0) + min(20, abs(skew) * 8)
    return Insight(
        category="Répartitions clés",
        title=f"Statistiques de « {col} »",
        narrative=narrative,
        importance=importance,
        fig=fig,
        table=desc.to_frame(name=col),
        icon="📈",
    )


def _is_metadata_col(col: str) -> bool:
    col_l = _norm(col)
    return any(k in col_l for k in METADATA_KEYWORDS)


def _is_redundant_pair(df: pd.DataFrame, a: str, b: str) -> bool:
    """
    Détecte les paires redondantes/tautologiques : cas fréquent avec les questions à choix
    multiples exportées de Kobo, où une colonne "indicateur" (ex: 'orylus', 'ir841') porte
    le nom d'une des modalités d'une autre colonne catégorielle (ex: la variété choisie).
    """
    try:
        a_name = _norm(a).replace("_", " ").strip()
        b_name = _norm(b).replace("_", " ").strip()
        vals_a = set(str(v).strip().lower() for v in df[a].dropna().unique())
        vals_b = set(str(v).strip().lower() for v in df[b].dropna().unique())
        if len(b_name) >= 3 and any(b_name in v or v in b_name for v in vals_a):
            return True
        if len(a_name) >= 3 and any(a_name in v or v in a_name for v in vals_b):
            return True
    except Exception:
        pass
    return False


def _insight_numeric_correlations(df: pd.DataFrame, numeric_cols: list, min_n: int = 30) -> list[Insight]:
    insights = []
    if len(numeric_cols) < 2:
        return insights
    corr = df[numeric_cols].corr(numeric_only=True)
    pairs = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            n_joint = df[[a, b]].dropna().shape[0]
            if n_joint < min_n:
                continue
            try:
                _, p_val = stats.pearsonr(*df[[a, b]].dropna().values.T)
            except Exception:
                p_val = 1.0
            if abs(r) >= 0.3 and p_val < 0.05:
                pairs.append((a, b, r, p_val))
    pairs.sort(key=lambda x: -abs(x[2]))

    if len(numeric_cols) >= 3:
        fig_hm = px.imshow(
            corr.round(2), text_auto=True, color_continuous_scale=DIVERGING,
            zmin=-1, zmax=1, title="Matrice de corrélation des variables numériques",
            aspect="auto",
        )
        fig_hm = style_fig(fig_hm, height=max(360, 40 * len(numeric_cols)))
        insights.append(Insight(
            category="Relations entre variables",
            title="Corrélations entre variables numériques",
            narrative=(
                "La matrice ci-dessous synthétise les corrélations linéaires (coefficient de Pearson) "
                "entre toutes les variables numériques. Les valeurs proches de +1 (vert) indiquent une "
                "relation positive forte, celles proches de -1 (rouge) une relation négative forte."
            ),
            importance=70,
            fig=fig_hm,
            icon="🔗",
        ))

    for a, b, r, p_val in pairs[:3]:
        strength = "forte" if abs(r) > 0.7 else "modérée"
        direction = "positive" if r > 0 else "négative"
        narrative = (
            f"Il existe une corrélation **{strength} {direction}** (r = {r:.2f}, statistiquement significative) "
            f"entre `{a}` et `{b}` : "
            + ("lorsque l'une augmente, l'autre a tendance à augmenter aussi."
               if r > 0 else "lorsque l'une augmente, l'autre a tendance à diminuer.")
        )
        fig = px.scatter(
            df, x=a, y=b, trendline="ols", opacity=0.6,
            title=f"{a} vs {b} (r = {r:.2f})",
            color_discrete_sequence=[PRIMARY],
        )
        fig = style_fig(fig, height=380)
        insights.append(Insight(
            category="Relations entre variables",
            title=f"Lien entre « {a} » et « {b} »",
            narrative=narrative,
            importance=60 + abs(r) * 30,
            fig=fig,
            icon="🔗",
        ))
    return insights


def _insight_cat_numeric(df: pd.DataFrame, categorical_cols: list, numeric_cols: list) -> list[Insight]:
    insights = []
    candidates = []
    for cat in categorical_cols:
        if _is_metadata_col(cat):
            continue
        nun = df[cat].nunique(dropna=True)
        if not (2 <= nun <= 8):
            continue
        for num in numeric_cols:
            eta, p_val, n = correlation_ratio(df[cat], pd.to_numeric(df[num], errors="coerce"))
            if eta >= 0.25 and p_val < 0.05:
                candidates.append((cat, num, eta, p_val))
    candidates.sort(key=lambda x: -x[2])

    for cat, num, eta, p_val in candidates[:3]:
        grp = df.groupby(cat)[num].mean(numeric_only=True).sort_values(ascending=False)
        best, worst = grp.index[0], grp.index[-1]
        narrative = (
            f"La valeur moyenne de `{num}` varie de façon statistiquement significative selon `{cat}` "
            f"(eta = {eta:.2f}). Le groupe **« {best} »** affiche la moyenne la plus élevée "
            f"({grp.iloc[0]:.2f}), contre **« {worst} »** la plus basse ({grp.iloc[-1]:.2f})."
        )
        fig = px.box(
            df, x=cat, y=num, color=cat,
            title=f"{num} selon {cat}",
            color_discrete_sequence=PALETTE,
        )
        fig = style_fig(fig, height=400, show_legend=False)
        insights.append(Insight(
            category="Relations entre variables",
            title=f"« {num} » selon « {cat} »",
            narrative=narrative,
            importance=55 + eta * 35,
            fig=fig,
            icon="⚖️",
        ))
    return insights


def _insight_cat_cat(df: pd.DataFrame, categorical_cols: list) -> list[Insight]:
    insights = []
    candidates = []
    cols = [c for c in categorical_cols if not _is_metadata_col(c) and 2 <= df[c].nunique(dropna=True) <= 10]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                ct = pd.crosstab(df[a], df[b])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                if _is_redundant_pair(df, a, b):
                    continue
                v, p_val, n = cramers_v(ct)
                if v >= 0.25 and p_val < 0.05:
                    candidates.append((a, b, v, ct))
            except Exception:
                continue
    candidates.sort(key=lambda x: -x[2])

    for a, b, v, ct in candidates[:2]:
        strength = "forte" if v > 0.5 else "modérée"
        narrative = (
            f"Une association **{strength}** et statistiquement significative (V de Cramér = {v:.2f}) est "
            f"observée entre `{a}` et `{b}` : la répartition de l'une dépend de l'autre "
            "(voir le détail des combinaisons ci-dessous)."
        )
        fig = px.imshow(
            ct, text_auto=True, color_continuous_scale=SEQ_GREEN,
            title=f"Tableau croisé — {a} × {b}", aspect="auto",
        )
        fig = style_fig(fig, height=max(360, 34 * ct.shape[0]))
        insights.append(Insight(
            category="Relations entre variables",
            title=f"Association « {a} » × « {b} »",
            narrative=narrative,
            importance=55 + v * 35,
            fig=fig,
            table=ct,
            icon="🧩",
        ))
    return insights


def _insight_temporal(df: pd.DataFrame, datetime_cols: list) -> list[Insight]:
    insights = []
    for col in datetime_cols[:1]:  # une seule variable temporelle principale suffit en général
        s = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dropna()
        if len(s) < 5:
            continue
        by_day = s.dt.date.value_counts().sort_index()
        span_days = (s.max() - s.min()).days
        peak_day = by_day.idxmax()
        narrative = (
            f"Les données de `{col}` couvrent une période de **{span_days} jour(s)**, "
            f"du {s.min().strftime('%d/%m/%Y')} au {s.max().strftime('%d/%m/%Y')}. "
            f"Le pic d'activité a été enregistré le **{peak_day.strftime('%d/%m/%Y')}** "
            f"avec {by_day.max()} enregistrement(s)."
        )
        weekday_counts = s.dt.day_name().value_counts()
        if len(weekday_counts) > 1:
            top_weekday = weekday_counts.idxmax()
            narrative += f" Le jour de la semaine le plus actif est **{top_weekday}**."

        fig = px.line(
            x=by_day.index, y=by_day.values, markers=True,
            labels={"x": "Date", "y": "Nombre d'enregistrements"},
            title=f"Évolution dans le temps — {col}",
            color_discrete_sequence=[PRIMARY],
        )
        fig.update_traces(fill="tozeroy", fillcolor="rgba(27,94,32,0.12)")
        fig = style_fig(fig, height=380)
        insights.append(Insight(
            category="Tendances temporelles",
            title=f"Évolution — {col}",
            narrative=narrative,
            importance=65,
            fig=fig,
            icon="📅",
        ))
    return insights


def _insight_geo(df: pd.DataFrame, profile: DataProfile) -> list[Insight]:
    insights = []
    if profile.geo_lat and profile.geo_lon:
        sub = df[[profile.geo_lat, profile.geo_lon]].dropna()
        sub = sub[sub[profile.geo_lat].between(-90, 90) & sub[profile.geo_lon].between(-180, 180)]
        if len(sub) >= 3:
            fig = px.scatter_mapbox(
                sub, lat=profile.geo_lat, lon=profile.geo_lon, zoom=6, height=460,
                color_discrete_sequence=[PRIMARY], opacity=0.7,
            )
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
            fig.update_layout(title="Localisation des enregistrements")
            insights.append(Insight(
                category="Répartition géographique",
                title="Carte des enregistrements",
                narrative=(
                    f"**{len(sub)} enregistrements géolocalisés** sont représentés sur la carte "
                    "ci-dessous, ce qui permet d'identifier visuellement les zones de forte et de "
                    "faible couverture de la collecte."
                ),
                importance=68,
                fig=fig,
                icon="🗺️",
            ))
    for col in profile.geo_categorical_cols[:2]:
        vc = df[col].dropna().astype(str).value_counts().head(15).sort_values(ascending=True)
        if len(vc) < 2:
            continue
        fig = px.bar(
            x=vc.values, y=vc.index, orientation="h",
            title=f"Répartition par {col}",
            labels={"x": "Nombre d'enregistrements", "y": ""},
            color=vc.values, color_continuous_scale=SEQ_GREEN,
        )
        fig.update_layout(coloraxis_showscale=False)
        fig = style_fig(fig, height=max(320, 30 * len(vc)))
        top_zone = vc.index[-1]
        top_share = vc.iloc[-1] / vc.sum() * 100
        insights.append(Insight(
            category="Répartition géographique",
            title=f"Couverture par « {col} »",
            narrative=(
                f"La zone **« {top_zone} »** concentre le plus grand nombre d'enregistrements "
                f"(**{_fmt_pct(top_share)}** du total), ce qui peut indiquer soit une couverture de "
                "collecte plus intense, soit une population/zone plus représentée."
            ),
            importance=52,
            fig=fig,
            icon="🗺️",
        ))
    return insights


# =============================================================================
# INSIGHTS ORIENTÉS OBJECTIF DE LA COLLECTE (contexte, indicateurs clés, zones)
# =============================================================================
def _rate_narrative(label_col: str, positive_share: float, positive_label: str) -> str:
    return f"**{_fmt_pct(positive_share)}** des répondants sont dans le cas « {positive_label} » pour « {label_col} »."


def _insight_context(df: pd.DataFrame, profile: DataProfile, ctx: SurveyContext) -> list[Insight]:
    """Génère la synthèse de contexte + un tableau d'indicateurs clés (KPI) directement
    exploitables, calculés à partir des rôles sémantiques détectés dans les données."""
    insights = []
    roles = ctx.roles
    kpi_rows = []  # (libellé, valeur affichée)
    narrative_bits = []

    # --- Genre ---
    for col in roles.get("gender", [])[:1]:
        vc = df[col].dropna().astype(str).value_counts()
        if len(vc) >= 2:
            top_label, top_val = vc.index[0], vc.iloc[0] / vc.sum() * 100
            kpi_rows.append((f"Répartition « {col} »", f"{top_label} : {_fmt_pct(top_val)}"))
            narrative_bits.append(
                f"la population enquêtée compte **{_fmt_pct(top_val)}** de « {top_label} » (variable `{col}`)"
            )

    # --- Âge ---
    for col in roles.get("age", [])[:1]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0:
            kpi_rows.append((f"Âge moyen ({col})", f"{s.mean():.1f} ans"))
            narrative_bits.append(f"l'âge moyen est de **{s.mean():.1f} ans** (médiane {s.median():.0f} ans)")

    # --- Expérience ---
    for col in roles.get("experience", [])[:1]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0:
            kpi_rows.append((f"Expérience moyenne ({col})", f"{s.mean():.1f} ans"))

    # --- Superficie ---
    for col in roles.get("surface", [])[:1]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0:
            kpi_rows.append((f"Superficie moyenne ({col})", f"{s.mean():.2f} ha (médiane {s.median():.2f})"))

    # --- Adhésion coopérative / organisation ---
    for col in roles.get("cooperative_membership", [])[:1]:
        vc = df[col].dropna().astype(str).str.lower().value_counts()
        yes_share = sum(v for k, v in vc.items() if k in BOOL_TRUE_SET) / vc.sum() * 100 if vc.sum() else 0
        kpi_rows.append((f"Membres d'une coopérative ({col})", _fmt_pct(yes_share)))
        narrative_bits.append(f"**{_fmt_pct(yes_share)}** des enquêtés sont membres d'une coopérative/organisation")

    # --- Accès à l'eau ---
    for col in roles.get("water_access", [])[:1]:
        vc = df[col].dropna().astype(str).str.lower().value_counts()
        yes_share = sum(v for k, v in vc.items() if k in BOOL_TRUE_SET) / vc.sum() * 100 if vc.sum() else 0
        kpi_rows.append((f"Accès à l'eau sur l'exploitation ({col})", _fmt_pct(yes_share)))

    # --- Niveau d'instruction ---
    for col in roles.get("education", [])[:1]:
        vc = df[col].dropna().astype(str).value_counts()
        if len(vc) > 0:
            kpi_rows.append((f"Niveau d'instruction dominant ({col})", f"{vc.index[0]} ({_fmt_pct(vc.iloc[0]/vc.sum()*100)})"))

def _make_gauge_figure(gauge_items: list) -> go.Figure:
    """Construit une mini-figure à jauges (0-100%) pour 2 à 4 indicateurs de taux clés."""
    n = len(gauge_items)
    fig = go.Figure()
    domain_width = 1.0 / n
    for idx, (label, value) in enumerate(gauge_items):
        x0 = idx * domain_width + 0.02
        x1 = (idx + 1) * domain_width - 0.02
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 26, "color": PRIMARY}},
            title={"text": label, "font": {"size": 12}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0.5},
                "bar": {"color": PRIMARY, "thickness": 0.35},
                "bgcolor": "#F1F8E9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#FFEBEE"},
                    {"range": [40, 70], "color": "#FFF8E1"},
                    {"range": [70, 100], "color": "#E8F5E9"},
                ],
            },
            domain={"x": [x0, x1], "y": [0, 1]},
        ))
    fig.update_layout(
        template="plotly_white", height=230,
        margin=dict(l=20, r=20, t=50, b=10),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _insight_context(df: pd.DataFrame, profile: DataProfile, ctx: SurveyContext) -> list[Insight]:
    """Génère la synthèse de contexte + des indicateurs clés (KPI) directement exploitables,
    calculés à partir des rôles sémantiques détectés dans les données, avec des visuels
    (jauges, donut) en plus du tableau récapitulatif."""
    insights = []
    roles = ctx.roles
    kpi_rows = []       # (libellé, valeur affichée) — pour le tableau
    gauge_items = []    # (libellé court, valeur %) — pour les jauges
    narrative_bits = []
    gender_col, gender_vc = None, None

    # --- Genre ---
    for col in roles.get("gender", [])[:1]:
        vc = df[col].dropna().astype(str).value_counts()
        if len(vc) >= 2:
            top_label, top_val = vc.index[0], vc.iloc[0] / vc.sum() * 100
            kpi_rows.append((f"Répartition « {col} »", f"{top_label} : {_fmt_pct(top_val)}"))
            narrative_bits.append(
                f"la population enquêtée compte **{_fmt_pct(top_val)}** de « {top_label} » (variable `{col}`)"
            )
            gender_col, gender_vc = col, vc

    # --- Âge ---
    for col in roles.get("age", [])[:1]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0:
            kpi_rows.append((f"Âge moyen ({col})", f"{s.mean():.1f} ans"))
            narrative_bits.append(f"l'âge moyen est de **{s.mean():.1f} ans** (médiane {s.median():.0f} ans)")

    # --- Expérience ---
    for col in roles.get("experience", [])[:1]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0:
            kpi_rows.append((f"Expérience moyenne ({col})", f"{s.mean():.1f} ans"))

    # --- Superficie ---
    for col in roles.get("surface", [])[:1]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0:
            kpi_rows.append((f"Superficie moyenne ({col})", f"{s.mean():.2f} ha (médiane {s.median():.2f})"))

    # --- Adhésion coopérative / organisation ---
    for col in roles.get("cooperative_membership", [])[:1]:
        vc = df[col].dropna().astype(str).str.lower().value_counts()
        yes_share = sum(v for k, v in vc.items() if k in BOOL_TRUE_SET) / vc.sum() * 100 if vc.sum() else 0
        kpi_rows.append((f"Membres d'une coopérative ({col})", _fmt_pct(yes_share)))
        narrative_bits.append(f"**{_fmt_pct(yes_share)}** des enquêtés sont membres d'une coopérative/organisation")
        gauge_items.append(("Membres coopérative", round(yes_share, 1)))

    # --- Accès à l'eau ---
    for col in roles.get("water_access", [])[:1]:
        vc = df[col].dropna().astype(str).str.lower().value_counts()
        yes_share = sum(v for k, v in vc.items() if k in BOOL_TRUE_SET) / vc.sum() * 100 if vc.sum() else 0
        kpi_rows.append((f"Accès à l'eau sur l'exploitation ({col})", _fmt_pct(yes_share)))
        gauge_items.append(("Accès à l'eau", round(yes_share, 1)))

    # --- Site aménagé ---
    for col in roles.get("site_amenagement", [])[:1]:
        vc = df[col].dropna().astype(str).str.lower().value_counts()
        yes_share = sum(v for k, v in vc.items() if k in BOOL_TRUE_SET) / vc.sum() * 100 if vc.sum() else 0
        kpi_rows.append((f"Site aménagé ({col})", _fmt_pct(yes_share)))
        gauge_items.append(("Site aménagé", round(yes_share, 1)))

    # --- Niveau d'instruction ---
    for col in roles.get("education", [])[:1]:
        vc = df[col].dropna().astype(str).value_counts()
        if len(vc) > 0:
            kpi_rows.append((f"Niveau d'instruction dominant ({col})", f"{vc.index[0]} ({_fmt_pct(vc.iloc[0]/vc.sum()*100)})"))

    theme_narrative = ctx.theme_label
    if narrative_bits:
        theme_narrative += " Concrètement : " + "; ".join(narrative_bits) + "."

    table = pd.DataFrame(kpi_rows, columns=["Indicateur", "Valeur"]) if kpi_rows else None

    # --- Visuel principal : jauges des taux clés (jusqu'à 4), sinon graphique à barres ---
    fig = None
    if len(gauge_items) >= 2:
        fig = _make_gauge_figure(gauge_items[:4])
    elif table is not None and len(table) >= 2:
        fig = px.bar(
            x=[1] * len(table), y=table["Indicateur"], orientation="h",
            title="Indicateurs clés", labels={"x": "", "y": ""},
        )
        fig = style_fig(fig, height=max(280, 40 * len(table)))

    insights.append(Insight(
        category="Contexte & indicateurs clés",
        title="Comprendre cette collecte",
        narrative=theme_narrative,
        importance=1000,  # toujours en tête
        fig=fig,
        table=table,
        icon="🎯",
    ))

    # --- Visuel complémentaire : donut de la répartition par genre ---
    if gender_col is not None and gender_vc is not None and len(gender_vc) >= 2:
        fig_gender = px.pie(
            names=gender_vc.index, values=gender_vc.values, hole=0.55,
            title=f"Répartition — {gender_col}",
            color_discrete_sequence=PALETTE,
        )
        fig_gender.update_traces(textinfo="percent+label")
        fig_gender = style_fig(fig_gender, height=360, show_legend=False)
        insights.append(Insight(
            category="Contexte & indicateurs clés",
            title=f"Répartition par genre ({gender_col})",
            narrative=(
                f"Sur {int(gender_vc.sum())} réponses valides, la répartition par genre est : "
                + ", ".join(f"**{k}** ({_fmt_pct(v/gender_vc.sum()*100)})" for k, v in gender_vc.items()) + "."
            ),
            importance=990,
            fig=fig_gender,
            icon="🎯",
        ))

    # --- Visuel complémentaire : niveau d'instruction ---
    for col in roles.get("education", [])[:1]:
        vc = df[col].dropna().astype(str).value_counts().head(10).sort_values(ascending=True)
        if len(vc) >= 2:
            fig_edu = px.bar(
                x=vc.values, y=vc.index, orientation="h",
                title=f"Niveau d'instruction — {col}",
                labels={"x": "Nombre de répondants", "y": ""},
                color=vc.values, color_continuous_scale=SEQ_GREEN, text=vc.values,
            )
            fig_edu.update_traces(textposition="outside", cliponaxis=False)
            fig_edu.update_layout(coloraxis_showscale=False)
            fig_edu = style_fig(fig_edu, height=max(300, 32 * len(vc)))
            insights.append(Insight(
                category="Contexte & indicateurs clés",
                title=f"Niveau d'instruction ({col})",
                narrative=(
                    f"La modalité la plus fréquente est **« {vc.index[-1]} »** "
                    f"({_fmt_pct(vc.iloc[-1] / vc.sum() * 100)} des répondants)."
                ),
                importance=980,
                fig=fig_edu,
                icon="🎯",
            ))

    return insights


def _insight_yield(df: pd.DataFrame, ctx: SurveyContext, zone_col: Optional[str]) -> list[Insight]:
    """Calcule un indicateur de rendement (production / surface), absent des données brutes
    mais essentiel pour toute analyse agricole exploitable, et le compare par zone si possible."""
    insights = []
    for prod_col, surf_col, marker in ctx.yield_pairs:
        try:
            sub = df[[prod_col, surf_col]].copy()
            sub.columns = ["prod", "surf"]
            sub = sub.apply(pd.to_numeric, errors="coerce").dropna()
            sub = sub[sub["surf"] > 0]
            if len(sub) < 15:
                continue
            sub["rendement_kg_ha"] = (sub["prod"] * 1000) / sub["surf"]
            # Retirer les valeurs extrêmes irréalistes (probables erreurs de saisie) au-delà du 99e percentile
            cap = sub["rendement_kg_ha"].quantile(0.99)
            sub_clean = sub[sub["rendement_kg_ha"] <= cap]

            period_label = {
                "cette_année": "cette année", "cette_annee": "cette année", "en_cours": "en cours",
                "actuel": "actuelle", "passée": "l'année passée", "passee": "l'année passée",
                "lannée_passée": "l'année passée", "lannee_passee": "l'année passée",
            }.get(marker, "")
            title_suffix = f" ({period_label})" if period_label else ""

            mean_y = sub_clean["rendement_kg_ha"].mean()
            median_y = sub_clean["rendement_kg_ha"].median()
            narrative = (
                f"À partir de `{prod_col}` et `{surf_col}`, le rendement moyen calculé{title_suffix} est "
                f"d'environ **{mean_y:,.0f} kg/ha** (médiane : {median_y:,.0f} kg/ha), sur "
                f"{len(sub_clean)} exploitations avec des données exploitables."
            ).replace(",", " ")

            fig = px.histogram(
                sub_clean, x="rendement_kg_ha", nbins=25,
                title=f"Rendement calculé{title_suffix} (kg/ha)",
                color_discrete_sequence=[PRIMARY],
                labels={"rendement_kg_ha": "Rendement (kg/ha)"},
            )
            fig.add_vline(x=mean_y, line_dash="dash", line_color="#EF5350",
                           annotation_text="Moyenne", annotation_position="top")
            fig = style_fig(fig, height=380)

            insights.append(Insight(
                category="Contexte & indicateurs clés",
                title=f"Rendement calculé{title_suffix}",
                narrative=narrative,
                importance=950,
                fig=fig,
                icon="🌾",
            ))

            # Comparaison du rendement par zone si une colonne de zone existe
            if zone_col and zone_col in df.columns:
                sub_zone = sub_clean.copy()
                sub_zone[zone_col] = df.loc[sub_zone.index, zone_col]
                sub_zone = sub_zone.dropna(subset=[zone_col])
                grp = sub_zone.groupby(zone_col)["rendement_kg_ha"].agg(["mean", "count"])
                grp = grp[grp["count"] >= 5].sort_values("mean", ascending=False)
                if len(grp) >= 2:
                    best, worst = grp.index[0], grp.index[-1]
                    ecart = grp["mean"].iloc[0] - grp["mean"].iloc[-1]
                    narrative_zone = (
                        f"Le rendement moyen varie sensiblement selon `{zone_col}` : la zone "
                        f"**« {best} »** obtient le meilleur rendement moyen ({grp['mean'].iloc[0]:,.0f} kg/ha), "
                        f"contre **« {worst} »** le plus faible ({grp['mean'].iloc[-1]:,.0f} kg/ha), soit un "
                        f"écart de **{ecart:,.0f} kg/ha**. Cet écart peut orienter le ciblage de l'appui "
                        "technique (semences, encadrement, accès à l'eau) vers les zones les moins performantes."
                    ).replace(",", " ")
                    fig_zone = px.bar(
                        x=grp["mean"].values, y=grp.index.astype(str), orientation="h",
                        title=f"Rendement moyen{title_suffix} par {zone_col}",
                        labels={"x": "Rendement moyen (kg/ha)", "y": ""},
                        color=grp["mean"].values, color_continuous_scale=SEQ_GREEN,
                        text=[f"{v:,.0f}".replace(",", " ") for v in grp["mean"].values],
                    )
                    fig_zone.update_traces(textposition="outside", cliponaxis=False)
                    fig_zone.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                    fig_zone = style_fig(fig_zone, height=max(320, 40 * len(grp)))
                    insights.append(Insight(
                        category="Comparaisons territoriales",
                        title=f"Rendement par zone{title_suffix} ({zone_col})",
                        narrative=narrative_zone,
                        importance=920,
                        fig=fig_zone,
                        table=grp.rename(columns={"mean": "Rendement moyen (kg/ha)", "count": "N"}),
                        icon="🏘️",
                    ))
        except Exception:
            continue
    return insights


def _insight_zone_comparison(df: pd.DataFrame, profile: DataProfile, ctx: SurveyContext) -> list[Insight]:
    """Compare les indicateurs numériques et les taux clés entre zones géographiques —
    la lecture territoriale demandée pour rendre l'analyse exploitable sur le terrain."""
    insights = []
    zone_col = ctx.zone_col
    if not zone_col or zone_col not in df.columns:
        return insights
    n_zones = df[zone_col].nunique(dropna=True)
    if not (2 <= n_zones <= 12):
        return insights

    # --- Comparaison des variables numériques clés par zone ---
    numeric_targets = []
    for role in ["surface", "experience", "age", "cooperative_size"]:
        numeric_targets += ctx.roles.get(role, [])
    numeric_targets = list(dict.fromkeys(numeric_targets))[:3]  # dédoublonner, limiter

    for num_col in numeric_targets:
        try:
            sub = df[[zone_col, num_col]].copy()
            sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
            sub = sub.dropna()
            grp = sub.groupby(zone_col)[num_col].agg(["mean", "count"])
            grp = grp[grp["count"] >= 5].sort_values("mean", ascending=False)
            if len(grp) < 2:
                continue
            best, worst = grp.index[0], grp.index[-1]
            narrative = (
                f"La moyenne de `{num_col}` varie selon `{zone_col}` : **« {best} »** affiche la valeur la "
                f"plus élevée ({grp['mean'].iloc[0]:.2f}), contre **« {worst} »** la plus faible "
                f"({grp['mean'].iloc[-1]:.2f})."
            )
            fig = px.bar(
                x=grp["mean"].values, y=grp.index.astype(str), orientation="h",
                title=f"{num_col} — comparaison par {zone_col}",
                labels={"x": "Moyenne", "y": ""},
                color=grp["mean"].values, color_continuous_scale=SEQ_GREEN,
            )
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            fig = style_fig(fig, height=max(320, 40 * len(grp)))
            insights.append(Insight(
                category="Comparaisons territoriales",
                title=f"« {num_col} » par {zone_col}",
                narrative=narrative,
                importance=88,
                fig=fig,
                table=grp.rename(columns={"mean": "Moyenne", "count": "N"}),
                icon="🏘️",
            ))
        except Exception:
            continue

    # --- Comparaison des taux clés (%) par zone : genre, coopérative, accès à l'eau ---
    rate_roles = ["gender", "cooperative_membership", "water_access", "site_amenagement", "cluster_membership"]
    for role in rate_roles:
        for cat_col in ctx.roles.get(role, [])[:1]:
            try:
                sub = df[[zone_col, cat_col]].dropna()
                if sub[cat_col].nunique() != 2:
                    continue
                # Modalité "positive" : la première valeur triée par fréquence globale
                positive_val = sub[cat_col].astype(str).value_counts().index[0]
                rate = sub.assign(is_pos=(sub[cat_col].astype(str) == positive_val)).groupby(zone_col)["is_pos"].mean() * 100
                counts = sub.groupby(zone_col).size()
                rate = rate[counts >= 5].sort_values(ascending=False)
                if len(rate) < 2:
                    continue
                best, worst = rate.index[0], rate.index[-1]
                narrative = (
                    f"La part de « {positive_val} » pour `{cat_col}` varie fortement selon `{zone_col}` : "
                    f"**{_fmt_pct(rate.iloc[0])}** en **« {best} »**, contre seulement "
                    f"**{_fmt_pct(rate.iloc[-1])}** en **« {worst} »**."
                )
                fig = px.bar(
                    x=rate.values, y=rate.index.astype(str), orientation="h",
                    title=f"Part de « {positive_val} » ({cat_col}) par {zone_col}",
                    labels={"x": "%", "y": ""},
                    color=rate.values, color_continuous_scale=SEQ_GREEN,
                    text=[f"{v:.0f}%" for v in rate.values],
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                fig = style_fig(fig, height=max(320, 40 * len(rate)))
                insights.append(Insight(
                    category="Comparaisons territoriales",
                    title=f"« {cat_col} » par {zone_col}",
                    narrative=narrative,
                    importance=85,
                    fig=fig,
                    icon="🏘️",
                ))
            except Exception:
                continue

    return insights



def generate_smart_insights(df: pd.DataFrame, profile: Optional[DataProfile] = None,
                              max_insights: int = 30) -> list[Insight]:
    """
    Point d'entrée principal : analyse le dataframe et retourne une liste
    d'Insight triés par pertinence (les plus importants en premier),
    plafonnée à `max_insights` pour rester lisible.
    """
    if profile is None:
        profile = profile_dataframe(df)

    ctx = build_survey_context(df, profile)

    insights: list[Insight] = []

    # --- 1. Comprendre la collecte : thème détecté + indicateurs clés ---
    insights += _insight_context(df, profile, ctx)

    # --- 2. Rendement calculé (agriculture) + comparaison par zone si pertinent ---
    insights += _insight_yield(df, ctx, ctx.zone_col)

    # --- 3. Comparaisons territoriales génériques (démographie, organisation...) ---
    insights += _insight_zone_comparison(df, profile, ctx)

    # --- 4. Qualité des données ---
    insights += _insight_data_quality(df, profile)

    # Colonnes déjà dotées de leur PROPRE visuel dédié dans le contexte (donut genre, barres
    # instruction, comparaisons par zone) : on évite seulement celles-là dans les répartitions
    # génériques ci-dessous, pour ne pas dupliquer un graphique identique. Les autres variables
    # de rôle (âge, superficie, coopérative, accès à l'eau...) gardent leur propre graphique
    # généré par la boucle générique, pour maximiser la richesse visuelle du rapport.
    covered_cols = set()
    covered_cols.update(ctx.roles.get("gender", []))
    covered_cols.update(ctx.roles.get("education", []))
    if ctx.zone_col:
        covered_cols.add(ctx.zone_col)

    # --- 5. Répartitions individuelles restantes (catégorielles + likert, plafonnées) ---
    cat_targets = [c for c in (profile.categorical_cols + profile.likert_cols) if c not in covered_cols][:10]
    for col in cat_targets:
        ins = _insight_categorical(df, col, "categorical")
        if ins:
            insights.append(ins)

    # --- 6. Statistiques numériques restantes (plafonnées) ---
    num_targets = [c for c in profile.numeric_cols if c not in covered_cols][:8]
    for col in num_targets:
        ins = _insight_numeric(df, col)
        if ins:
            insights.append(ins)

    # --- 7. Relations statistiques génériques (corrélations, associations significatives) ---
    insights += _insight_numeric_correlations(df, profile.numeric_cols)
    insights += _insight_cat_numeric(df, profile.categorical_cols + profile.likert_cols, profile.numeric_cols)
    insights += _insight_cat_cat(df, profile.categorical_cols)

    # --- 8. Tendances temporelles ---
    insights += _insight_temporal(df, profile.datetime_cols)

    # --- 9. Répartition géographique générale (carte, couverture par zone) ---
    insights += _insight_geo(df, profile)

    # Tri : contexte/indicateurs clés et qualité des données toujours en tête,
    # le reste trié par pertinence décroissante.
    priority = [i for i in insights if i.category in ("Contexte & indicateurs clés", "Qualité des données")]
    priority.sort(key=lambda i: -i.importance)
    others = [i for i in insights if i.category not in ("Contexte & indicateurs clés", "Qualité des données")]
    others.sort(key=lambda i: -i.importance)

    return (priority + others)[:max_insights]


def build_executive_summary(df: pd.DataFrame, profile: DataProfile, insights: list[Insight]) -> str:
    """Génère un court résumé exécutif en langage naturel à partir des insights les plus importants."""
    lines = [
        f"Ce jeu de données comprend **{profile.n_rows:,}".replace(",", " ") + f" enregistrements** "
        f"répartis sur **{profile.n_cols} variables**, avec un taux de complétude global de "
        f"**{_fmt_pct(profile.completeness)}**."
    ]
    top = [i for i in insights if i.category != "Qualité des données"][:4]
    if top:
        lines.append("Principaux constats :")
        for ins in top:
            # Première phrase de la narration, simplifiée
            first_sentence = re.split(r"(?<=[.!?])\s", ins.narrative.replace("**", ""))[0]
            lines.append(f"- {first_sentence}")
    return "\n".join(lines)


# =============================================================================
# RENDU STREAMLIT
# =============================================================================
def _kpi_row(profile: DataProfile):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enregistrements", f"{profile.n_rows:,}".replace(",", " "))
    c2.metric("Variables", profile.n_cols)
    c3.metric("Complétude", _fmt_pct(profile.completeness))
    c4.metric("Doublons", profile.duplicate_rows)


def render_smart_dashboard(df: pd.DataFrame, meta: Optional[dict] = None) -> list[Insight]:
    """
    Affiche le tableau de bord intelligent dans Streamlit et retourne
    la liste des insights générés (pour réutilisation, ex. export du rapport).
    """
    if not _HAS_STREAMLIT:
        raise RuntimeError("streamlit n'est pas disponible dans cet environnement.")

    if df is None or df.empty:
        st.info("Aucune donnée chargée. Chargez d'abord un jeu de données pour lancer l'analyse intelligente.")
        return []

    with st.spinner("🧠 Analyse intelligente du jeu de données en cours..."):
        profile = profile_dataframe(df)
        insights = generate_smart_insights(df, profile)

    st.markdown("### 🧭 Vue d'ensemble")
    _kpi_row(profile)

    summary = build_executive_summary(df, profile, insights)
    with st.container(border=True):
        st.markdown("#### 📝 Résumé exécutif")
        st.markdown(summary)

    st.session_state["_smart_insights_cache"] = insights
    st.session_state["_smart_profile_cache"] = profile

    categories = []
    for ins in insights:
        if ins.category not in categories:
            categories.append(ins.category)

    if not categories:
        st.warning("Aucun insight n'a pu être généré pour ce jeu de données.")
        return insights

    tabs = st.tabs([f"{c}" for c in categories])
    for tab, cat in zip(tabs, categories):
        with tab:
            cat_insights = [i for i in insights if i.category == cat]
            for ins in cat_insights:
                with st.container(border=True):
                    st.markdown(f"##### {ins.icon} {ins.title}")
                    st.markdown(ins.narrative)
                    if ins.fig is not None:
                        st.plotly_chart(ins.fig, use_container_width=True)
                    if ins.table is not None:
                        with st.expander("📋 Voir le détail chiffré"):
                            st.dataframe(ins.table, use_container_width=True)

    return insights
