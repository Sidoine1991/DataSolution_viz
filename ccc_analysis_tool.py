#!/usr/bin/env python
# =============================================================================
# ccc_analysis_tool.py  —  Pipeline d'analyse avancée des collectes KoboToolbox
# =============================================================================
# Permet d'analyser en profondeur les données de la baseline FIPA et autres
# collectes : statistiques univariées, bivariées, multivariées, interprétation
# par IA, visualisations, export.
#
# Utilisation :
#   python ccc_analysis_tool.py analyze --url <URL> --ai-provider claude_free
#   python ccc_analysis_tool.py healthcheck
#
# Intégration dashboard :
#   from ccc_analysis_tool import FIPAAnalysisPipeline
#   pipeline = FIPAAnalysisPipeline(url="...", ai_provider=...)
#   result = pipeline.run_full_analysis()
# =============================================================================

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pickle
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
console = Console()

# --- Constantes --------------------------------------------------------------

DEFAULT_FIPA_URL = (
    "https://kf.kobotoolbox.org/private-media/AnonymousUser/exports/"
    "arSNAHn25xfwT34PZoo7iE/"
    "Enqu%C3%AAte_Baseline_FIPA__CCR-B__Rikolto_B%C3%A9nin_-_all_versions_-_default_-_2026-07-20-16-36-07.xlsx"
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "analysis.sqlite"

PIPELINE_CONFIG = {
    "cache_ttl_hours": 24,
    "max_rows_memory": 5000,
    "alpha_significance": 0.05,
    "force_rerun": False,
}

# --- Énumérations -----------------------------------------------------------

class AnalysisStage(str, Enum):
    LOAD = "chargement"
    CLEAN = "nettoyage"
    UNIVARIATE = "univariée"
    BIVARIATE = "bivariée"
    MULTIVARIATE = "multivariée"
    AI_INTERPRET = "interprétation IA"
    VISUALIZE = "visualisation"
    PERSIST = "persistance"


class AIProvider(str, Enum):
    CLAUDE_FREE = "claude_free"
    OPENAI = "openai"
    CLAUDE_API = "claude_api"
    DISABLE = "disable"


def provider_available(p: AIProvider) -> bool:
    if p == AIProvider.CLAUDE_FREE:
        try:
            from claudecodefree import ClaudeCode
            return True
        except Exception:
            return False
    if p == AIProvider.OPENAI:
        try:
            import openai
            return bool(os.getenv("OPENAI_API_KEY"))
        except Exception:
            return False
    if p == AIProvider.CLAUDE_API:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return True


# --- Dataclasses ------------------------------------------------------------

@dataclass
class DataQualityReport:
    total_rows: int
    total_cols: int
    missing_pct: float
    duplicate_rows: int
    issues: List[str] = field(default_factory=list)

@dataclass
class UnivariateFinding:
    column: str
    col_type: str  # numeric / categorical
    dominant_value: Optional[str] = None
    dominant_pct: Optional[float] = None
    unique_count: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

@dataclass
class BivariateFinding:
    var_x: str
    var_y: str
    test_type: str  # chi2 / anova / correlation
    statistic: float = 0.0
    p_value: float = 1.0
    effect_size: Optional[float] = None
    interpretation: Optional[str] = None

@dataclass
class MultivariateFinding:
    technique: str  # correlation_matrix / pca / ancova / kmeans
    components: Optional[List[str]] = None
    explained_variance: Optional[List[float]] = None
    n_clusters: Optional[int] = None
    insights: Optional[str] = None

@dataclass
class AnalysisResult:
    stage: AnalysisStage
    timestamp: datetime = field(default_factory=datetime.now)
    data_quality: Optional[DataQualityReport] = None
    univariate: List[UnivariateFinding] = field(default_factory=list)
    bivariate: List[BivariateFinding] = field(default_factory=list)
    multivariate: List[MultivariateFinding] = field(default_factory=list)
    ai_insights: Optional[str] = None
    figures: Dict[str, Any] = field(default_factory=dict)

# --- Pipeline principal -----------------------------------------------------

class FIPAAnalysisPipeline:
    """Pipeline complet d'analyse pour les données de collecte KoboToolbox."""

    def __init__(
        self,
        url: str = DEFAULT_FIPA_URL,
        ai_provider: AIProvider = AIProvider.DISABLE,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        db_path: Path = DB_PATH,
        cache_ttl_hours: int = PIPELINE_CONFIG["cache_ttl_hours"],
    ):
        self.url = url
        self.ai_provider = ai_provider
        self.api_key = api_key or os.getenv("AI_API_KEY")
        self.enable_cache = enable_cache
        self.db_path = db_path
        self.cache_ttl_hours = cache_ttl_hours
        self._init_db()

    # ------------------------------------------------------------------
    # Base de données locale
    # ------------------------------------------------------------------
    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE,
                stage TEXT,
                timestamp TEXT,
                data_quality TEXT,
                univariate TEXT,
                bivariate TEXT,
                multivariate TEXT,
                ai_insights TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ck ON sessions(cache_key)")
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cache_key(self, suffix: str = "") -> str:
        return hashlib.sha256(f"{self.url}:{suffix}".encode()).hexdigest()[:16]

    def _cache_get(self, key: str) -> Optional[Dict]:
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.execute("SELECT * FROM sessions WHERE cache_key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "stage": row[3],
            "timestamp": row[4],
            "data_quality": json.loads(row[5]) if row[5] else None,
            "univariate": json.loads(row[6]) if row[6] else None,
            "bivariate": json.loads(row[7]) if row[7] else None,
            "multivariate": json.loads(row[8]) if row[8] else None,
            "ai_insights": row[9],
        }

    def _cache_put(self, key: str, result: AnalysisResult):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT OR REPLACE INTO sessions
            (cache_key, stage, timestamp, data_quality, univariate,
             bivariate, multivariate, ai_insights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            key,
            result.stage.value,
            result.timestamp.isoformat(),
            json.dumps(asdict(result.data_quality)) if result.data_quality else None,
            json.dumps([asdict(u) for u in result.univariate]),
            json.dumps([asdict(b) for b in result.bivariate]),
            json.dumps([asdict(m) for m in result.multivariate]),
            result.ai_insights,
        ))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Chargement des données
    # ------------------------------------------------------------------
    def load_data(self, force: bool = False) -> Optional[pd.DataFrame]:
        ck = self._cache_key("raw_df")
        cache_file = CACHE_DIR / f"{ck}.pkl"

        if not force and self.enable_cache and cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(hours=self.cache_ttl_hours):
                logger.info(f"Cache HIT : {cache_file.name}")
                return pd.read_pickle(cache_file)

        logger.info(f"Téléchargement depuis {self.url}")
        resp = requests.get(self.url, timeout=120)
        resp.raise_for_status()

        if "xlsx" in resp.headers.get("content-type", "") or self.url.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(resp.content))
        else:
            df = pd.read_csv(io.BytesIO(resp.content))

        df = self._normalize(df)
        self._cache_df(df, ck)
        logger.info(f"Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().replace("\n", " ").replace("\r", "") for c in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass
        return df

    def _cache_df(self, df: pd.DataFrame, key: str):
        path = CACHE_DIR / f"{key}.pkl"
        df.to_pickle(path)

    # ------------------------------------------------------------------
    # Qualité des données
    # ------------------------------------------------------------------
    def assess_quality(self, df: pd.DataFrame) -> DataQualityReport:
        n, p = df.shape
        missing = int(df.isnull().sum().sum())
        missing_pct = (missing / (n * p)) * 100 if n > 0 else 0.0
        dupes = int(df.duplicated().sum())
        issues = []
        if missing_pct > 20:
            issues.append(f"Fort taux de valeurs manquantes ({missing_pct:.1f}%)")
        if dupes > n * 0.05:
            issues.append(f"Doublons excessifs ({dupes})")
        if n == 0:
            issues.append("Aucune donnée")
        return DataQualityReport(
            total_rows=n, total_cols=p,
            missing_pct=round(missing_pct, 2),
            duplicate_rows=dupes,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Analyse univariée
    # ------------------------------------------------------------------
    def run_univariate(self, df: pd.DataFrame) -> List[UnivariateFinding]:
        findings = []
        for col in df.columns:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            if pd.api.types.is_numeric_dtype(vals):
                findings.append(UnivariateFinding(
                    column=col,
                    col_type="numeric",
                    unique_count=int(vals.nunique()),
                    mean=float(vals.mean()),
                    std=float(vals.std()),
                    min_val=float(vals.min()),
                    max_val=float(vals.max()),
                    skewness=float(vals.skew()),
                    kurtosis=float(vals.kurtosis()),
                ))
            else:
                vc = vals.value_counts(normalize=True)
                top = vc.index[0]
                top_pct = float(vc.iloc[0] * 100)
                findings.append(UnivariateFinding(
                    column=col,
                    col_type="categorical",
                    dominant_value=str(top),
                    dominant_pct=round(top_pct, 2),
                    unique_count=int(vals.nunique()),
                ))
        return findings

    # ------------------------------------------------------------------
    # Analyse bivariée
    # ------------------------------------------------------------------
    def run_bivariate(self, df: pd.DataFrame) -> List[BivariateFinding]:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        findings = []

        for c1, c2 in self._pairs(num_cols[:6]):
            mask = df[[c1, c2]].dropna()
            if len(mask) < 10:
                continue
            r, p = stats.pearsonr(mask[c1], mask[c2])
            findings.append(BivariateFinding(
                var_x=c1, var_y=c2, test_type="correlation",
                statistic=round(r, 4), p_value=round(p, 4),
            ))

        for cat in cat_cols[:3]:
            for num in num_cols[:3]:
                groups = [g[num].dropna() for _, g in df.groupby(cat) if len(g[num].dropna()) > 1]
                if len(groups) >= 2:
                    f, p = stats.f_oneway(*groups)
                    findings.append(BivariateFinding(
                        var_x=cat, var_y=num, test_type="anova",
                        statistic=round(f, 4), p_value=round(p, 4),
                    ))

        for c1, c2 in self._pairs(cat_cols[:4]):
            ct = pd.crosstab(df[c1], df[c2])
            if ct.size > 0 and ct.min().min() > 0:
                chi2, p, dof, _ = stats.chi2_contingency(ct)
                findings.append(BivariateFinding(
                    var_x=c1, var_y=c2, test_type="chi2",
                    statistic=round(chi2, 4), p_value=round(p, 4),
                ))

        return findings

    @staticmethod
    def _pairs(items: List[str]) -> List[Tuple[str, str]]:
        return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]

    # ------------------------------------------------------------------
    # Analyse multivariée
    # ------------------------------------------------------------------
    def run_multivariate(self, df: pd.DataFrame) -> Tuple[List[MultivariateFinding], Dict]:
        findings = []
        figs = {}
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return findings, figs

        corr = num.corr()
        figs["correlation_heatmap"] = px.imshow(
            corr, text_auto=".2f", title="Matrice de corrélation",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        strong = sum(1 for i in range(len(corr)) for j in range(i+1, len(corr)) if abs(corr.iloc[i, j]) > 0.5)
        findings.append(MultivariateFinding(
            technique="correlation_matrix",
            insights=f"{strong} paires fortement corrélées (|r| > 0.5)",
        ))

        if num.shape[1] >= 5:
            scaler = StandardScaler()
            # Imputation robuste : mediane pour colonnes, sinon 0
            num_imputed = num.fillna(num.median())
            # Si toujours des NaN (colonne vide), remplacer par 0
            num_imputed = num_imputed.fillna(0)
            scaled = scaler.fit_transform(num_imputed)
            pca = PCA(n_components=min(3, num.shape[1]))
            comps = pca.fit_transform(scaled)
            var = pca.explained_variance_ratio_
            figs["pca"] = px.scatter(
                x=comps[:, 0], y=comps[:, 1],
                title="PCA — Composantes principales 1 vs 2",
                labels={"x": f"PC1 ({var[0]:.1%})", "y": f"PC2 ({var[1]:.1%})"},
            )
            findings.append(MultivariateFinding(
                technique="pca",
                components=[f"PC{i+1}" for i in range(len(var))],
                explained_variance=[round(v, 4) for v in var],
                insights=f"Les 2 premières composantes expliquent {sum(var[:2]):.1%} de la variance",
            ))

        if num.shape[1] >= 3:
            scaler = StandardScaler()
            num_imputed = num.fillna(num.median()).fillna(0)
            scaled = scaler.fit_transform(num_imputed)
            inertias = []
            for k in range(2, min(8, num.shape[0])):
                inertias.append(KMeans(n_clusters=k, n_init=10).fit(scaled).inertia_)
            figs["elbow"] = px.line(
                x=list(range(2, min(8, num.shape[0]))), y=inertias,
                title="Méthode Elbow pour K-Means",
                labels={"x": "k (clusters)", "y": "Inertie"},
            )

        return findings, figs

    # ------------------------------------------------------------------
    # Interprétation IA
    # ------------------------------------------------------------------
    def generate_ai_insights(self, df: pd.DataFrame, result: AnalysisResult) -> str:
        if self.ai_provider == AIProvider.DISABLE:
            return "Interprétation IA désactivée."

        summary = (
            f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes\n"
            f"Colonnes : {df.columns.tolist()}\n"
            f"Types : {dict(df.dtypes.astype(str))}\n"
            f"Analyses univariées : {len(result.univariate)} variables\n"
            f"Analyses bivariées : {len(result.bivariate)} paires testées\n"
            f"Analyses multivariées : {len(result.multivariate)} techniques"
        )

        prompt = f"""Tu es un expert en analyse de données de projet agricole.

Contexte de l'analyse :
{summary}

Interprète les résultats en français :
1. Points clés et tendances principales
2. Relations importantes entre variables
3. Recommandations opérationnelles
4. Limites à prendre en compte

Réponse concise (max 500 mots)."""

        try:
            if self.ai_provider == AIProvider.CLAUDE_FREE:
                from claudecodefree import ClaudeCode
                client = ClaudeCode()
                return client.chat(prompt) or "Aucune réponse IA."
            elif self.ai_provider == AIProvider.OPENAI:
                import openai
                openai.api_key = self.api_key
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                )
                return resp.choices[0].message.content
            else:
                return "Fournisseur IA non supporté dans ce contexte."
        except Exception as e:
            logger.warning(f"Erreur IA : {e}")
            return f"Impossible de générer l'interprétation IA ({e})."

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    def create_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        figs = {}
        for col in df.select_dtypes(include=[np.number]).columns[:5]:
            figs[f"hist_{col}"] = px.histogram(
                df, x=col, title=f"Distribution de {col}",
                marginal="box", opacity=0.7,
            )
        for col in df.select_dtypes(exclude=[np.number]).columns[:3]:
            vc = df[col].value_counts().reset_index()
            vc.columns = [col, "count"]
            figs[f"bar_{col}"] = px.bar(
                vc, x=col, y="count", title=f"Fréquence de {col}",
            )
        return figs

    # ------------------------------------------------------------------
    # Pipeline complet
    # ------------------------------------------------------------------
    def run_full_analysis(
        self,
        include_ai: bool = True,
        verbose: bool = False,
    ) -> AnalysisResult:
        result = AnalysisResult(stage=AnalysisStage.LOAD)
        ck = self._cache_key("full")

        if self.enable_cache:
            cached = self._cache_get(ck)
            if cached:
                logger.info("Résultat complet en cache, restitution.")
                result.stage = AnalysisStage(cached["stage"])
                result.timestamp = datetime.fromisoformat(cached["timestamp"])
                if cached["data_quality"]:
                    result.data_quality = DataQualityReport(**cached["data_quality"])
                result.univariate = [UnivariateFinding(**u) for u in (cached["univariate"] or [])]
                result.bivariate = [BivariateFinding(**b) for b in (cached["bivariate"] or [])]
                result.multivariate = [MultivariateFinding(**m) for m in (cached["multivariate"] or [])]
                result.ai_insights = cached["ai_insights"]
                return result

        df = self.load_data()
        if df is None:
            raise RuntimeError("Impossible de charger les données.")

        result.data_quality = self.assess_quality(df)
        result.univariate = self.run_univariate(df)
        if verbose:
            console.print(f"[green]✓ Univariée : {len(result.univariate)} variables[/]")

        result.bivariate = self.run_bivariate(df)
        if verbose:
            console.print(f"[green]✓ Bivariée : {len(result.bivariate)} paires[/]")

        mv, mv_figs = self.run_multivariate(df)
        result.multivariate = mv
        result.figures.update(mv_figs)
        if verbose:
            console.print(f"[green]✓ Multivariée : {len(mv)} techniques[/]")

        if include_ai and self.ai_provider != AIProvider.DISABLE:
            result.ai_insights = self.generate_ai_insights(df, result)
            if verbose:
                console.print("[green]✓ Interprétation IA générée[/]")

        viz_figs = self.create_visualizations(df)
        result.figures.update(viz_figs)
        result.stage = AnalysisStage.PERSIST

        self._cache_put(ck, result)
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_json(self, result: AnalysisResult, path: Optional[Path] = None) -> Path:
        path = path or DATA_DIR / f"analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
        data = {
            "timestamp": result.timestamp.isoformat(),
            "data_quality": asdict(result.data_quality) if result.data_quality else None,
            "univariate": [asdict(u) for u in result.univariate],
            "bivariate": [asdict(b) for b in result.bivariate],
            "multivariate": [asdict(m) for m in result.multivariate],
            "ai_insights": result.ai_insights,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def export_figures_html(self, result: AnalysisResult, path: Optional[Path] = None) -> Path:
        path = path or DATA_DIR / f"figures_{datetime.now():%Y%m%d_%H%M%S}.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'></head><body>\n")
            f.write("<h1>Graphiques — Analyse FIPA</h1>\n")
            for name, fig in result.figures.items():
                f.write(f"<h2>{name}</h2>\n")
                if hasattr(fig, "to_html"):
                    f.write(fig.to_html(full_html=False))
                elif hasattr(fig, "to_plotly_json"):
                    f.write(go.Figure(fig).to_html(full_html=False))
            f.write("</body></html>\n")
        return path


# --- CLI ---------------------------------------------------------------------

app = typer.Typer(help="Pipeline d'analyse avancée des collectes KoboToolbox")


@app.command()
def analyze(
    url: str = typer.Option(DEFAULT_FIPA_URL, "--url", help="URL du fichier Excel"),
    ai_provider: str = typer.Option("disable", "--ai", help="Fournisseur IA (disable, claude_free, openai)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Dossier de sortie"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Mode verbeux"),
):
    """Analyse complète des données de collecte."""
    try:
        provider = AIProvider(ai_provider.lower())
    except ValueError:
        console.print(f"[red]Fournisseur IA invalide : {ai_provider}[/]")
        raise typer.Exit(1)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler()],
    )

    console.print(f"[bold blue]Analyse des données[/]")
    console.print(f"  URL : {url}")
    console.print(f"  IA  : {provider.value}")
    console.print()

    pipeline = FIPAAnalysisPipeline(url=url, ai_provider=provider)
    result = pipeline.run_full_analysis(include_ai=provider != AIProvider.DISABLE, verbose=verbose)

    out_dir = output or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = pipeline.export_json(result, out_dir / "analysis_result.json")
    html_path = pipeline.export_figures_html(result, out_dir / "figures.html")

    console.print(f"\n[green]Analyse terminée ![/]")
    console.print(f"  Fichier JSON : {json_path}")
    console.print(f"  Graphiques   : {html_path}")
    if result.ai_insights:
        console.print(f"\n[bold]Interprétation IA :[/]\n{result.ai_insights[:500]}...")


@app.command()
def healthcheck():
    """Vérifie la configuration et les dépendances."""
    console.print("[bold]Healthcheck du pipeline[/]")
    for name in ["pandas", "numpy", "scipy", "statsmodels", "plotly",
                  "sklearn", "requests", "typer", "rich"]:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "N/A")
            console.print(f"  [green]✓ {name} {ver}[/]")
        except Exception:
            console.print(f"  [red]✗ {name} manquant[/]")

    console.print(f"\nDossier de cache : {CACHE_DIR} ({'OK' if CACHE_DIR.exists() else 'à créer'})")
    console.print(f"Base de données : {DB_PATH}")
    console.print(f"URL par défaut   : {DEFAULT_FIPA_URL[:80]}...")

    for p in [AIProvider.CLAUDE_FREE, AIProvider.OPENAI, AIProvider.CLAUDE_API]:
        status = "disponible" if provider_available(p) else "non configuré"
        console.print(f"  {p.value:15s} : {status}")


@app.command()
def cache_clear():
    """Vide le cache d'analyse."""
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM sessions")
    conn.commit()
    conn.close()
    console.print("[green]Cache vidé[/]")


if __name__ == "__main__":
    app()
