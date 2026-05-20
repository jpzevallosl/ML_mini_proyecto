"""
Mini Proyecto #1: Clasificacion de Cardiotocografia Fetal
=========================================================
Universidad de Ingenieria y Tecnologia (UTEC) - Inteligencia Artificial

REPRODUCIBILIDAD:
    - Semilla fija: np.random.seed(42), random_state=42 en sklearn
    - Tiempo de ejecucion: ~3-5 minutos en CPU moderna (1 core)
    - Sin GPU requerida

REQUISITOS (instalar antes de ejecutar):
    pip install scikit-learn pandas numpy matplotlib seaborn

USO:
    1. Estructura esperada del directorio:
        ctg_mini_proyecto/
        |-- cardiotocographic.csv   (dataset)
        |-- code/
        |   |-- run_experiments.py  (este archivo)
        |-- results/                (se crea automaticamente)
        |-- figures/                (se crea automaticamente)

    2. Desde la raiz del proyecto, ejecutar:
        python3 code/run_experiments.py

SALIDAS GENERADAS:
    - results/grid_search_full.csv      Todas las combinaciones de hiperparametros
    - results/per_model_metrics.csv     Mejor configuracion + metricas por clase
    - results/summary.json              Experimento completo en JSON
    - figures/class_distribution.pdf    Distribucion de clases
    - figures/correlation.pdf           Matriz de correlacion de features
    - figures/model_comparison.pdf      Comparacion F1-macro entre modelos
    - figures/confusion_matrix.pdf      Matriz de confusion del mejor modelo

Autor: Estudiante UTEC
Fecha: 2026-05-15
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# ============================================================
# Semilla (NumPy seed según se solicita en el enunciado)
# ============================================================
SEED = 42
np.random.seed(SEED)

# Rutas relativas a la ubicacion del script (portable)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))  # carpeta raiz del proyecto
OUT = os.path.join(ROOT, 'results')
FIGS = os.path.join(ROOT, 'figures')
DATA = os.path.join(ROOT, 'cardiotocographic.csv')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# ============================================================
# 1. CARGA Y PREPROCESAMIENTO
# ============================================================
print('=' * 70)
print('1. CARGA Y PREPROCESAMIENTO')
print('=' * 70)

df = pd.read_csv(DATA)
print(f'Dataset: {df.shape[0]} registros, {df.shape[1]} columnas')
print(f'Clases:\n{df["CLASE"].value_counts().sort_index()}')

X = df.drop(columns=['CLASE']).values
y = df['CLASE'].values
feat_names = df.drop(columns=['CLASE']).columns.tolist()

# Split estratificado 85/15 (≈300 muestras de test interno)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)
print(f'\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}')

# ============================================================
# 2. EDA: Figuras descriptivas
# ============================================================
print('\n' + '=' * 70)
print('2. ANÁLISIS EXPLORATORIO')
print('=' * 70)

# Distribución de clases
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
labels = ['Normal (1)', 'Sospechoso (2)', 'Patológico (3)']
counts = df['CLASE'].value_counts().sort_index().values
pcts = 100 * counts / counts.sum()
bars = ax.bar(labels, counts, color=['#2c7bb6', '#fdae61', '#d7191c'])
for b, c, p in zip(bars, counts, pcts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15,
            f'{c} ({p:.1f}%)', ha='center', fontsize=9)
ax.set_ylabel('Número de registros')
ax.set_title('Distribución de clases (n = 1998)')
ax.set_ylim(0, max(counts) * 1.15)
plt.tight_layout()
plt.savefig(f'{FIGS}/class_distribution.pdf', bbox_inches='tight')
plt.close()

# Matriz de correlación
fig, ax = plt.subplots(figsize=(8, 7))
corr = df.drop(columns=['CLASE']).corr()
sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=feat_names, yticklabels=feat_names,
            cbar_kws={'shrink': 0.7}, ax=ax, square=True, linewidths=0.3)
ax.set_title('Matriz de correlación de Pearson entre features')
plt.tight_layout()
plt.savefig(f'{FIGS}/correlation.pdf', bbox_inches='tight')
plt.close()
print('Figuras EDA generadas')

# ============================================================
# 3. CONFIGURACIÓN DE MODELOS Y GRIDS
# ============================================================
print('\n' + '=' * 70)
print('3. MODELOS Y GRIDS DE HIPERPARÁMETROS')
print('=' * 70)

# Pipelines: escalado para LogReg/SVM/KNN; sin escalado para árbol
pipelines = {
    'LogReg': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, random_state=SEED,
                                   solver='lbfgs'))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(random_state=SEED, probability=False))
    ]),
    'DecisionTree': Pipeline([
        ('clf', DecisionTreeClassifier(random_state=SEED))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ]),
}

param_grids = {
    'LogReg': {
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__class_weight': [None, 'balanced'],
    },
    'SVM': {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__kernel': ['linear', 'rbf'],
        'clf__gamma': ['scale', 0.1],
        'clf__class_weight': ['balanced'],
    },
    'DecisionTree': {
        'clf__max_depth': [None, 5, 10, 20],
        'clf__min_samples_split': [2, 5, 10],
        'clf__class_weight': [None, 'balanced'],
    },
    'KNN': {
        'clf__n_neighbors': [3, 5, 7, 11, 15],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan'],
    },
}

# ============================================================
# 4. K-FOLD CV + GRID SEARCH (selección de hiperparámetros)
# ============================================================
print('\n' + '=' * 70)
print('4. K-FOLD CV (5 folds) + GRID SEARCH')
print('=' * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
all_grid_results = {}
best_estimators = {}

for name in pipelines:
    print(f'\n--- {name} ---')
    t0 = time.time()
    gs = GridSearchCV(pipelines[name], param_grids[name],
                      scoring='f1_macro', cv=cv, n_jobs=-1,
                      return_train_score=False)
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f'Tiempo: {elapsed:.1f}s | Combinaciones: {len(gs.cv_results_["mean_test_score"])}')
    print(f'Mejor F1-macro CV: {gs.best_score_:.4f}')
    print(f'Mejores params: {gs.best_params_}')

    # Guardar todo el grid para tabla del paper
    cvr = gs.cv_results_
    rows = []
    for i in range(len(cvr['mean_test_score'])):
        rows.append({
            'modelo': name,
            'params': str(cvr['params'][i]),
            'f1_macro_cv': cvr['mean_test_score'][i],
            'f1_macro_std': cvr['std_test_score'][i],
            'rank': cvr['rank_test_score'][i],
        })
    all_grid_results[name] = rows
    best_estimators[name] = gs.best_estimator_

# ============================================================
# 5. PRECISION / RECALL / F1 POR CLASE en CV para mejores hiperparámetros
# ============================================================
print('\n' + '=' * 70)
print('5. MÉTRICAS POR CLASE EN CV (mejores hiperparámetros)')
print('=' * 70)

cv_metrics = {}
classes = [1, 2, 3]
class_names = ['Normal', 'Sospechoso', 'Patologico']

for name, est in best_estimators.items():
    # Métricas CV manual para tener por clase
    per_fold = {f'{m}_{c}': [] for m in ['prec', 'rec', 'f1'] for c in classes}
    per_fold.update({'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                     'accuracy': []})

    for tr_idx, val_idx in cv.split(X_train, y_train):
        Xt, Xv = X_train[tr_idx], X_train[val_idx]
        yt, yv = y_train[tr_idx], y_train[val_idx]
        est.fit(Xt, yt)
        yp = est.predict(Xv)
        per_fold['accuracy'].append(accuracy_score(yv, yp))
        per_fold['prec_macro'].append(precision_score(yv, yp, average='macro',
                                                      zero_division=0))
        per_fold['rec_macro'].append(recall_score(yv, yp, average='macro',
                                                  zero_division=0))
        per_fold['f1_macro'].append(f1_score(yv, yp, average='macro',
                                             zero_division=0))
        p = precision_score(yv, yp, average=None, labels=classes, zero_division=0)
        r = recall_score(yv, yp, average=None, labels=classes, zero_division=0)
        f = f1_score(yv, yp, average=None, labels=classes, zero_division=0)
        for i, c in enumerate(classes):
            per_fold[f'prec_{c}'].append(p[i])
            per_fold[f'rec_{c}'].append(r[i])
            per_fold[f'f1_{c}'].append(f[i])

    cv_metrics[name] = {k: (float(np.mean(v)), float(np.std(v)))
                        for k, v in per_fold.items()}
    print(f'\n{name}:')
    print(f'  F1-macro: {cv_metrics[name]["f1_macro"][0]:.4f} ± {cv_metrics[name]["f1_macro"][1]:.4f}')
    print(f'  Accuracy: {cv_metrics[name]["accuracy"][0]:.4f} ± {cv_metrics[name]["accuracy"][1]:.4f}')

# ============================================================
# 6. BOOTSTRAP (500 réplicas) para estimación robusta del error
# ============================================================
print('\n' + '=' * 70)
print('6. BOOTSTRAP .632 (500 réplicas)')
print('=' * 70)

N_BOOT = 500
rng = np.random.default_rng(SEED)
bootstrap_results = {}

for name, est in best_estimators.items():
    print(f'\n{name}: bootstrap...', end='', flush=True)
    t0 = time.time()
    f1_in_list, f1_out_list = [], []

    for b in range(N_BOOT):
        idx = rng.integers(0, len(X_train), len(X_train))
        oob_mask = np.ones(len(X_train), dtype=bool)
        oob_mask[np.unique(idx)] = False

        if oob_mask.sum() < 5:  # por si acaso
            continue
        Xb, yb = X_train[idx], y_train[idx]
        Xo, yo = X_train[oob_mask], y_train[oob_mask]
        est.fit(Xb, yb)
        yp_in = est.predict(Xb)
        yp_out = est.predict(Xo)
        f1_in_list.append(f1_score(yb, yp_in, average='macro', zero_division=0))
        f1_out_list.append(f1_score(yo, yp_out, average='macro', zero_division=0))

    err_in = 1 - np.mean(f1_in_list)
    err_out = 1 - np.mean(f1_out_list)
    err_632 = 0.368 * err_in + 0.632 * err_out  # .632 estimator
    f1_632 = 1 - err_632
    ci_low, ci_high = np.percentile(f1_out_list, [2.5, 97.5])

    bootstrap_results[name] = {
        'f1_in_mean': float(np.mean(f1_in_list)),
        'f1_oob_mean': float(np.mean(f1_out_list)),
        'f1_632': float(f1_632),
        'ci95_low': float(ci_low),
        'ci95_high': float(ci_high),
        'err_in': float(err_in),
        'err_oob': float(err_out),
        'err_632': float(err_632),
    }
    print(f' F1.632={f1_632:.4f} | OOB F1={np.mean(f1_out_list):.4f} '
          f'(CI95: [{ci_low:.4f}, {ci_high:.4f}]) | {time.time() - t0:.1f}s')

# ============================================================
# 7. ELECCIÓN DEL MEJOR MODELO Y EVALUACIÓN EN TEST
# ============================================================
print('\n' + '=' * 70)
print('7. MEJOR MODELO Y MATRIZ DE CONFUSIÓN EN TEST')
print('=' * 70)

# Criterio: F1-macro .632 más alto (combina training y OOB)
best_name = max(bootstrap_results, key=lambda k: bootstrap_results[k]['f1_632'])
print(f'\nMejor modelo (por F1.632): {best_name}')

# Re-entrenar con todo el train, evaluar en test
final_est = best_estimators[best_name]
final_est.fit(X_train, y_train)
y_pred = final_est.predict(X_test)

acc_test = accuracy_score(y_test, y_pred)
f1m_test = f1_score(y_test, y_pred, average='macro')
print(f'Accuracy en test: {acc_test:.4f}')
print(f'F1-macro en test: {f1m_test:.4f}')
print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred, target_names=class_names,
                            digits=4, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

# Figura matriz de confusión (porcentajes)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', cbar=True,
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            vmin=0, vmax=100, annot_kws={'size': 11})
ax.set_xlabel('Predicción')
ax.set_ylabel('Real')
ax.set_title(f'Matriz de confusión - {best_name} (test, %)\n'
             f'Accuracy = {acc_test * 100:.2f}%   F1-macro = {f1m_test:.4f}')
plt.tight_layout()
plt.savefig(f'{FIGS}/confusion_matrix.pdf', bbox_inches='tight')
plt.close()

# Figura comparativa de modelos
fig, ax = plt.subplots(figsize=(7, 4))
names_order = ['LogReg', 'SVM', 'DecisionTree', 'KNN']
f1_cv = [cv_metrics[n]['f1_macro'][0] for n in names_order]
f1_cv_std = [cv_metrics[n]['f1_macro'][1] for n in names_order]
f1_boot = [bootstrap_results[n]['f1_632'] for n in names_order]
f1_oob = [bootstrap_results[n]['f1_oob_mean'] for n in names_order]
x = np.arange(len(names_order))
w = 0.27
ax.bar(x - w, f1_cv, w, yerr=f1_cv_std, label='5-fold CV',
       color='#2c7bb6', capsize=3)
ax.bar(x, f1_boot, w, label='Bootstrap .632', color='#fdae61')
ax.bar(x + w, f1_oob, w, label='Bootstrap OOB', color='#d7191c')
ax.set_xticks(x)
ax.set_xticklabels(names_order)
ax.set_ylabel('F1-macro')
ax.set_title('Comparación de modelos: F1-macro por método de validación')
ax.legend(loc='lower right', fontsize=8)
ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/model_comparison.pdf', bbox_inches='tight')
plt.close()

# ============================================================
# 8. GUARDAR TODOS LOS RESULTADOS
# ============================================================
print('\n' + '=' * 70)
print('8. GUARDANDO RESULTADOS')
print('=' * 70)

# Tabla de hiperparámetros completa (todas las combinaciones)
all_grid_df = pd.concat([pd.DataFrame(v) for v in all_grid_results.values()],
                        ignore_index=True)
all_grid_df.to_csv(f'{OUT}/grid_search_full.csv', index=False)
print(f'  -> {OUT}/grid_search_full.csv')

# Tabla de mejores params + métricas por clase
best_rows = []
for name in names_order:
    m = cv_metrics[name]
    row = {
        'Modelo': name,
        'Mejores hiperparametros': str(best_estimators[name].get_params()),
        'F1_macro_CV_mean': m['f1_macro'][0],
        'F1_macro_CV_std': m['f1_macro'][1],
        'Accuracy_CV': m['accuracy'][0],
        'Precision_macro_CV': m['prec_macro'][0],
        'Recall_macro_CV': m['rec_macro'][0],
    }
    for c, cn in zip(classes, class_names):
        row[f'Precision_{cn}'] = m[f'prec_{c}'][0]
        row[f'Recall_{cn}'] = m[f'rec_{c}'][0]
        row[f'F1_{cn}'] = m[f'f1_{c}'][0]
    row['F1_macro_632'] = bootstrap_results[name]['f1_632']
    row['F1_OOB'] = bootstrap_results[name]['f1_oob_mean']
    row['CI95_low'] = bootstrap_results[name]['ci95_low']
    row['CI95_high'] = bootstrap_results[name]['ci95_high']
    best_rows.append(row)

best_df = pd.DataFrame(best_rows)
best_df.to_csv(f'{OUT}/per_model_metrics.csv', index=False)
print(f'  -> {OUT}/per_model_metrics.csv')

# JSON con todo para que el paper lo lea
summary = {
    'seed': SEED,
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'n_features': int(X_train.shape[1]),
    'class_counts': {int(k): int(v) for k, v in df['CLASE'].value_counts().items()},
    'best_model': best_name,
    'best_params_per_model': {n: best_estimators[n].get_params() for n in names_order},
    'cv_metrics': cv_metrics,
    'bootstrap': bootstrap_results,
    'test_accuracy': float(acc_test),
    'test_f1_macro': float(f1m_test),
    'confusion_matrix_counts': cm.tolist(),
    'confusion_matrix_pct': cm_pct.tolist(),
}
# Convertir params no serializables
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(x) for x in o]
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float64)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    try:
        json.dumps(o)
        return o
    except (TypeError, ValueError):
        return str(o)

with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(clean(summary), f, indent=2)
print(f'  -> {OUT}/summary.json')

print('\nDONE.')

# ============================================================
# 9. PREDICCIÓN SOBRE TEST EXTERNO (127 registros, pestaña 2)
# ============================================================
TEST_FILE = os.path.join(ROOT, 'test_127.csv')
if os.path.exists(TEST_FILE):
    print('\n' + '=' * 70)
    print('9. PREDICCIÓN SOBRE TEST EXTERNO (127 registros sin etiqueta)')
    print('=' * 70)

    df_ext = pd.read_csv(TEST_FILE)
    print(f'Test externo: {df_ext.shape[0]} registros, {df_ext.shape[1]} columnas')

    # Entrenar el mejor modelo con TODOS los 1998 registros
    final_full = best_estimators[best_name]
    final_full.fit(X, y)  # X, y son los 1998 completos
    X_ext = df_ext.values
    y_ext_pred = final_full.predict(X_ext)

    # Distribución de predicciones
    unique, counts = np.unique(y_ext_pred, return_counts=True)
    print(f'\nDistribución de predicciones del {best_name}:')
    label_map = {1: 'Normal', 2: 'Sospechoso', 3: 'Patológico'}
    for u, c in zip(unique, counts):
        print(f'  Clase {u} ({label_map[u]}): {c} ({100*c/len(y_ext_pred):.1f}%)')

    # Guardar CSV con predicciones
    df_pred = df_ext.copy()
    df_pred['CLASE_PREDICHA'] = y_ext_pred
    pred_path = os.path.join(OUT, 'predicciones_test_127.csv')
    df_pred.to_csv(pred_path, index=False)
    print(f'\n  -> {pred_path}')
    print(f'\nNota: el test externo NO tiene etiquetas reales. Las predicciones')
    print(f'      se entregan para evaluación externa por el profesor.')
else:
    print(f'\nNota: no se encontró {TEST_FILE}. Coloque el archivo test_127.csv')
    print(f'      en la raíz del proyecto para generar predicciones.')

