#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Versión que NO requiere argumentos:
- Si no se pasa --input_csv, usa 'input_benchmark.csv'.
- Si el CSV no existe, lo genera automáticamente.
"""

import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_INPUT = "input_benchmark.csv"
DEFAULT_OUTPUT = "resultados_benchmark_semantico.csv"


def compute_similarity(expected: str, generated: str) -> float:
    vect = TfidfVectorizer().fit([expected, generated])
    tfidf = vect.transform([expected, generated])
    sim = cosine_similarity(tfidf[0], tfidf[1])[0, 0]
    return float(sim)


def ensure_default_csv(path: str):
    """Si no existe el CSV, lo crea automáticamente."""
    if os.path.exists(path):
        return

    print(f"[INFO] No existe {path}. Generando CSV por defecto...")
    df = pd.DataFrame([
        {
            "id": 1,
            "query": "¿Cómo lograr una textura más suave en una crema?",
            "expected_answer": "Usar ingredientes ricos en proteínas como crema o leche para obtener suavidad."
        },
        {
            "id": 2,
            "query": "¿Qué pasa si cocino carne a baja temperatura?",
            "expected_answer": "La cocción lenta mantiene la humedad y deja la carne tierna."
        },
        {
            "id": 3,
            "query": "¿Para qué sirve tamizar la harina?",
            "expected_answer": "Sirve para airear la harina y eliminar grumos logrando masas esponjosas."
        }
    ])
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[INFO] Archivo creado: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark semántico sin necesidad de argumentos."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_INPUT,
        help="Archivo de entrada (por defecto input_benchmark.csv)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Archivo de salida (por defecto resultados_benchmark_semantico.csv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Si no existe el CSV, se genera automático
    ensure_default_csv(args.input_csv)

    # 2) Leer CSV
    df = pd.read_csv(args.input_csv)

    resultados = []

    for _, row in df.iterrows():
        query = row["query"]
        expected = row["expected_answer"]

        # ===========================================
        # AQUÍ VA TU VARIABLE REAL DE RESPUESTA
        # ===========================================
        response = f"Respuesta generada por mi modelo para: {query}"
        # ===========================================

        sim = compute_similarity(expected, response)

        resultados.append({
            "id": row["id"],
            "query": query,
            "expected_answer": expected,
            "generated_answer": response,
            "similarity": sim
        })

    df_out = pd.DataFrame(resultados)
    df_out.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\nResultados guardados en: {args.output_csv}")
    print(df_out[["id", "similarity"]])


if __name__ == "__main__":
    main()

