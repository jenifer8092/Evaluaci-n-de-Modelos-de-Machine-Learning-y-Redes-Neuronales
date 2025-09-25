#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inferencia con pipeline de sklearn + export de datos LIMPIOS.
Uso:
    python infer_pipeline.py --csv test_inferencia.csv --pkl random_forest_pipeline.pkl --out resultados_inferencia_limpio.csv

Opciones:
    --csv        Ruta al CSV de entrada (datos crudos).
    --pkl        Ruta al pipeline entrenado en .pkl (joblib).
    --out        Ruta del CSV de salida (resultados con DF limpio).
    --id-col     Nombre de la columna ID a preservar al frente si existe (default: ID).
    --head       Cuántas filas imprimir como preview (default: 10).
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import joblib

def get_cleaned_df_from_pipeline(pipeline, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve el DataFrame limpiado usando el paso 'cleaning' del pipeline.
    Si no existe, retorna una copia del DF original.
    """
    try:
        if hasattr(pipeline, "named_steps") and "cleaning" in pipeline.named_steps:
            cleaned = pipeline.named_steps["cleaning"].transform(df_raw)
            # Asegurar DataFrame
            if not isinstance(cleaned, pd.DataFrame):
                cleaned = pd.DataFrame(cleaned, index=df_raw.index)
            return cleaned
        else:
            return df_raw.copy()
    except Exception as e:
        print(f"[Aviso] No fue posible obtener el DataFrame limpio: {e}", file=sys.stderr)
        return df_raw.copy()

def main():
    parser = argparse.ArgumentParser(description="Inferencia con pipeline y export en base a DF limpio.")
    parser.add_argument("--csv", required=True, help="Ruta al CSV de entrada (datos crudos).")
    parser.add_argument("--pkl", required=True, help="Ruta al pipeline entrenado (.pkl).")
    parser.add_argument("--out", default="resultados_inferencia_limpio.csv", help="Ruta del CSV de salida.")
    parser.add_argument("--id-col", default="ID", help="Nombre de la columna ID a preservar (si existe).")
    parser.add_argument("--head", type=int, default=10, help="Filas a mostrar como vista previa.")
    args = parser.parse_args()

    # Validaciones rápidas
    if not os.path.exists(args.csv):
        print(f"\nArchivo '{args.csv}' no encontrado. Coloca el archivo en el directorio actual o indica la ruta correcta.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.pkl):
        print(f"\nArchivo de pipeline '{args.pkl}' no encontrado.", file=sys.stderr)
        sys.exit(1)

    try:
        # Cargar insumos
        test_data = pd.read_csv(args.csv)
        print("\n=== INFERENCIA EN ARCHIVO DE PRUEBA ===")
        print(f"Archivo de entrada: {args.csv}")
        print(f"Forma del archivo de prueba (crudo): {test_data.shape}")

        pipeline = joblib.load(args.pkl)

        # Obtener DF limpio (sólo para salida/auditoría)
        cleaned_test_data = get_cleaned_df_from_pipeline(pipeline, test_data)
        print(f"Forma del archivo de prueba (limpio): {cleaned_test_data.shape}")

        # Predicción usando TODO el pipeline (limpieza + prepro + modelo)
        predictions  = pipeline.predict(test_data)

        # Probabilidades (si el estimador las soporta)
        try:
            probabilities = pipeline.predict_proba(test_data)
            has_proba = True
        except Exception:
            probabilities = None
            has_proba = False
            print("[Aviso] El estimador no soporta predict_proba; no se guardarán probabilidades.")

        # Armar resultados sobre el DF LIMPIO
        results = cleaned_test_data.copy()
        # Conservar ID al frente si existe en el crudo
        if args.id_col in test_data.columns and args.id_col not in results.columns:
            results.insert(0, args.id_col, test_data[args.id_col].values)

        # Predicción
        results["Prediccion"] = predictions

        # Probabilidades (si existen)
        if has_proba:
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                results["Probabilidad_No_Default"] = probabilities[:, 0]
                results["Probabilidad_Default"]    = probabilities[:, 1]
            else:
                # Modelos con una sola probabilidad (raro) o decision_function
                results["Probabilidad"] = probabilities.ravel()

        # Guardar
        results.to_csv(args.out, index=False)
        print(f"Resultados (LIMPIO) guardados en '{args.out}'")

        # Métricas rápidas
        try:
            binc = np.bincount(predictions.astype(int))
            binc_str = binc
        except Exception:
            binc_str = pd.Series(predictions).value_counts().sort_index()
        print(f"Predicciones realizadas: {len(predictions)}")
        print(f"Distribución de predicciones: {binc_str}")

        # Mostrar algunas filas
        cols_show = ["Prediccion"]
        if has_proba and probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            cols_show += ["Probabilidad_Default"]
        if args.id_col in results.columns:
            cols_show = [args.id_col] + cols_show

        preview_cols = [c for c in cols_show if c in results.columns]
        print(f"\nPrimeras {args.head} filas:")
        if preview_cols:
            print(results[preview_cols].head(args.head))
        else:
            print(results.head(args.head))

    except FileNotFoundError as e:
        print(f"\nArchivo no encontrado: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError al procesar inferencia: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
