# Mini Proyecto #1: Clasificación de Cardiotocografía Fetal

## Contenido
- `paper/main.pdf` — Paper compilado (5 páginas, formato IEEE conference)
- `paper/main.tex` — Fuente LaTeX editable
- `figures/` — 4 figuras del paper (PDF vectorial)
- `results/grid_search_full.csv` — 64 combinaciones evaluadas con F1-macro CV
- `results/per_model_metrics.csv` — Mejor config + métricas por clase y bootstrap
- `results/summary.json` — Experimento completo en JSON
- `code/run_experiments.py` — Script reproducible (semilla 42)
- `cardiotocographic.csv` — Dataset parseado del PDF (1998 registros)

## Reproducir
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python3 code/run_experiments.py
```
Semilla: `np.random.seed(42)` y `random_state=42` en sklearn.

## Recompilar el paper
```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```
Necesita `texlive-publishers` para la clase IEEEtran.

## Resultados clave
- Dataset: 1998 CTG, 20 features, 3 clases (77.8% / 14.0% / 8.2%)
- Split estratificado: 1698 train / 300 test
- **Mejor modelo: Decision Tree** (`min_samples_split=10`)
  - F1-macro CV = 0.882 ± 0.025
  - F1-macro bootstrap .632 = 0.897
  - Accuracy test = 92.33% / F1-macro test = 0.878

## Notas
1. **Test set (127 registros)**: la pestaña 2 que menciona el enunciado no se incluyó en la entrega. Se usó split interno 85/15 (300 muestras) sobre los 1998 disponibles. Cuando reciban la pestaña 2, basta cargarla al final del script y predecir.
2. **1998 vs 1999**: el PDF tiene 1998 filas extraíbles; verifiquen contra el Excel original.
3. **GPU**: el código corre en CPU sin problema (~5 min). sklearn clásico no usa GPU; para acelerar con GPU usen cuML o XGBoost con `tree_method='gpu_hist'`.
4. **GitHub/Colab**: actualicen el placeholder `https://github.com/<usuario>/ctg-mini-proyecto` en el paper antes de entregar.
