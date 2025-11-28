#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_semantico_recetas.py

Script de ejemplo para evaluar un benchmark semántico simple
basado en preguntas y respuestas sobre recetas de alimentos.

- Lee un CSV con columnas: id, query, expected_answer.
- Genera (o carga) respuestas de un sistema/modelo.
- Calcula una medida de similitud entre la respuesta esperada
  y la respuesta generada usando TF-IDF + coseno.
- Guarda los resultados en un CSV de salida.

Este script es una plantilla para el curso de Proyecto Aplicado
del Magíster en Inteligencia Artificial.
"""

import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def generate_model_answer(query: str) -> str:
    """Genera una respuesta de ejemplo del 'modelo'.

    IMPORTANTE:
    - Esta función es un placeholder.
    - Aquí debes reemplazar por una llamada a tu modelo real
      (por ejemplo, un LLM, un sistema de recuperación, etc.).
    - Por defecto, devolvemos una respuesta muy simple para que
      el script sea ejecutable sin dependencias externas.
    """
    # TODO: Reemplazar por tu propio sistema
    # Por ahora, solo devolvemos una respuesta genérica:
    if "textura más suave" in query:
        return "Para una textura más suave se usan ingredientes con proteínas y buena cantidad de aminoácidos, como leche o crema."
    if "baja temperatura" in query:
        return "La cocción lenta a baja temperatura hace que la carne quede más tierna y conserve mejor la humedad."
    if "marinada" in query:
        return "El ácido de limón o vinagre ayuda a suavizar la carne y resaltar el sabor."
    if "tamizar la harina" in query:
        return "Tamizar la harina rompe grumos y agrega aire, logrando masas más esponjosas."
    if "emulsionante" in query or "mayonesa" in query:
        return "La yema de huevo actúa como emulsionante gracias a la lecitina, uniendo agua y grasa."
    # Respuesta genérica fallback
    return "Depende de la receta y de los ingredientes utilizados."


def compute_similarity(expected: str, generated: str) -> float:
    """Calcula similitud coseno entre dos textos usando TF-IDF."""
    vect = TfidfVectorizer().fit([expected, generated])
    tfidf = vect.transform([expected, generated])
    sim = cosine_similarity(tfidf[0], tfidf[1])[0, 0]
    return float(sim)


def run_benchmark(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        q = row["query"]
        expected = row["expected_answer"]
        generated = generate_model_answer(q)
        sim = compute_similarity(expected, generated)

        results.append({
            "id": row["id"],
            "query": q,
            "expected_answer": expected,
            "generated_answer": generated,
            "similarity": sim
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Resultados guardados en: {output_csv}")
    print(df_out[["id", "similarity"]])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark semántico de preguntas sobre recetas de alimentos."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Ruta al CSV con columnas: id, query, expected_answer."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="resultados_benchmark_semantico.csv",
        help="Ruta al CSV de salida con similitudes."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
