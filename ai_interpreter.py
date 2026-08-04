# -*- coding: utf-8 -*-
"""
ai_interpreter.py
==================
Couche d'interprétation IA, agnostique du fournisseur, branchée sur les
insights produits par smart_analysis.py. L'utilisateur fournit sa propre
clé API (OpenAI, Google Gemini, Anthropic Claude, ou tout endpoint
compatible OpenAI — Groq, Together, Ollama local, etc.) et obtient une
lecture stratégique en français : interprétation des résultats,
hypothèses d'explication, recommandations opérationnelles.

La clé API n'est jamais écrite sur disque : elle ne vit que dans
st.session_state, côté client de la session Streamlit en cours.
"""

from __future__ import annotations

import json
from typing import Optional

import requests

PROVIDERS = {
    "openai": "OpenAI (GPT)",
    "anthropic": "Anthropic (Claude)",
    "gemini": "Google (Gemini)",
    "openai_compatible": "Autre — endpoint compatible OpenAI (Groq, Together, Ollama...)",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5-20250929",
    "gemini": "gemini-2.0-flash",
    "openai_compatible": "llama-3.3-70b-versatile",
}


class AIInterpreterError(Exception):
    pass


def _call_openai(prompt: str, api_key: str, model: str, base_url: str = "https://api.openai.com/v1") -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Tu es un analyste MEAL (Suivi-Évaluation) expert des données de développement rural et agricole en Afrique de l'Ouest. Tu réponds en français, de façon structurée, concrète et actionnable."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 1800,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code >= 400:
        raise AIInterpreterError(f"Erreur API ({r.status_code}) : {r.text[:400]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1800,
        "system": "Tu es un analyste MEAL (Suivi-Évaluation) expert des données de développement rural et agricole en Afrique de l'Ouest. Tu réponds en français, de façon structurée, concrète et actionnable.",
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code >= 400:
        raise AIInterpreterError(f"Erreur API ({r.status_code}) : {r.text[:400]}")
    data = r.json()
    return "".join(block.get("text", "") for block in data.get("content", []))


def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1800},
    }
    r = requests.post(url, json=payload, timeout=90)
    if r.status_code >= 400:
        raise AIInterpreterError(f"Erreur API ({r.status_code}) : {r.text[:400]}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise AIInterpreterError(f"Réponse inattendue de l'API Gemini : {json.dumps(data)[:400]}")


def call_ai(provider: str, prompt: str, api_key: str, model: Optional[str] = None,
            base_url: Optional[str] = None) -> str:
    """Point d'entrée unique, quel que soit le fournisseur choisi."""
    if not api_key:
        raise AIInterpreterError("Aucune clé API fournie.")
    model = model or DEFAULT_MODELS.get(provider)
    if provider == "openai":
        return _call_openai(prompt, api_key, model)
    if provider == "anthropic":
        return _call_anthropic(prompt, api_key, model)
    if provider == "gemini":
        return _call_gemini(prompt, api_key, model)
    if provider == "openai_compatible":
        if not base_url:
            raise AIInterpreterError("Un endpoint (base_url) est requis pour un fournisseur compatible OpenAI.")
        return _call_openai(prompt, api_key, model, base_url=base_url)
    raise AIInterpreterError(f"Fournisseur inconnu : {provider}")


# =============================================================================
# CONSTRUCTION DU PROMPT À PARTIR DES INSIGHTS
# =============================================================================
def build_interpretation_prompt(profile, insights, theme_label: str, extra_context: str = "") -> str:
    """Construit un prompt compact (texte, pas de données brutes) à partir des insights déjà
    calculés par smart_analysis, pour demander à l'IA une interprétation approfondie."""
    lines = [
        "Voici la synthèse chiffrée d'une collecte de données (déjà analysée statistiquement).",
        f"Contexte détecté automatiquement : {theme_label}",
        f"Volume : {profile.n_rows} enregistrements, {profile.n_cols} variables, "
        f"{profile.completeness}% de complétude.",
        "",
        "Constats et indicateurs déjà calculés (par ordre de pertinence décroissante) :",
    ]
    for ins in insights[:22]:
        lines.append(f"- [{ins.category}] {ins.title} : {ins.narrative}")
        if ins.table is not None:
            try:
                small = ins.table.head(8)
                lines.append(f"  Détail chiffré : {small.to_dict()}")
            except Exception:
                pass

    if extra_context:
        lines.append("")
        lines.append(f"Contexte additionnel fourni par l'utilisateur : {extra_context}")

    lines += [
        "",
        "En te basant UNIQUEMENT sur ces constats (n'invente pas de chiffres non fournis), rédige "
        "une interprétation approfondie en français, structurée ainsi :",
        "1. **Synthèse générale** (3-4 phrases) : que retenir globalement de cette collecte ?",
        "2. **Constats saillants et leur portée opérationnelle** (5-7 points) : pour chaque constat "
        "important, explique ce qu'il signifie concrètement pour le programme et pourquoi c'est "
        "important (pas juste répéter le chiffre).",
        "3. **Disparités territoriales** : si des comparaisons par zone sont présentes, souligne "
        "les écarts les plus importants et leurs implications pour le ciblage des interventions.",
        "4. **Hypothèses explicatives prudentes** : pour 2-3 constats marquants, propose des pistes "
        "d'explication plausibles (à vérifier sur le terrain), sans les présenter comme certaines.",
        "5. **Recommandations opérationnelles concrètes** (4-6 points) : actions priorisées et "
        "réalistes pour une équipe MEAL/programme, formulées comme des décisions actionnables.",
        "6. **Limites de l'analyse** (2-3 points) : biais possibles, données manquantes, précautions "
        "d'interprétation.",
        "Reste factuel, évite le jargon inutile, et n'utilise que les informations fournies ci-dessus.",
    ]
    return "\n".join(lines)


def generate_ai_interpretation(profile, insights, theme_label: str, provider: str, api_key: str,
                                 model: Optional[str] = None, base_url: Optional[str] = None,
                                 extra_context: str = "") -> str:
    prompt = build_interpretation_prompt(profile, insights, theme_label, extra_context)
    return call_ai(provider, prompt, api_key, model=model, base_url=base_url)
