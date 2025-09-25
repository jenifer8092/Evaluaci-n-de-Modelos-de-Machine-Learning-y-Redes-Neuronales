Para ejecutar el script de test del Pipeline con el archivo de test_inferencia.csv

```bash
python infer_pipeline.py --csv test_inferencia.csv --pkl random_forest_pipeline.pkl --out resultados_inferencia_limpio.csv
```

| Flag       | Requerido | Descripción                                                              | Default                            |
| ---------- | --------- | ------------------------------------------------------------------------ | ---------------------------------- |
| `--csv`    | Sí        | Ruta al CSV de entrada (datos crudos).                                   | —                                  |
| `--pkl`    | Sí        | Ruta al pipeline entrenado (`.pkl`).                                     | —                                  |
| `--out`    | No        | Ruta del CSV de salida (datos limpios + predicciones).                   | `resultados_inferencia_limpio.csv` |
