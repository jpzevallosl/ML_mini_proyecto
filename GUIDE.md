# Guía de reproducción

Esta guía está pensada para que un evaluador (profesor o equipo revisor)
pueda reproducir todos los resultados del paper desde cero.

**Tiempo total estimado: 5-10 minutos** (1-2 min de instalación + 3-5 min de
ejecución).

---

## Paso 1 — Verificar Python

Necesitas Python 3.10 o superior. Verifica con:

```bash
python3 --version
```

Si no lo tienes:

- **Ubuntu / WSL**: `sudo apt install python3 python3-pip python3-venv`
- **macOS**: `brew install python` (o descarga desde https://python.org)
- **Windows**: descarga desde https://python.org y marca *"Add to PATH"*
  durante la instalación.

---

## Paso 2 — Descomprimir el proyecto

Descomprime el ZIP entregado. Deberías ver esta estructura:

```
ctg_mini_proyecto/
├── README.md
├── GUIDE.md                    ← esta guía
├── requirements.txt            ← dependencias
├── cardiotocographic.csv       ← dataset (1998 filas)
├── code/
│   └── run_experiments.py      ← script principal
├── paper/
│   ├── main.pdf                ← paper compilado
│   └── main.tex                ← fuente LaTeX
├── figures/                    ← 4 PDFs de figuras
└── results/                    ← CSVs y JSON con resultados
```

Abre una terminal dentro de `ctg_mini_proyecto/`.

---

## Paso 3 — Crear entorno virtual (recomendado)

Esto evita conflictos con otros paquetes del sistema.

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Cuando termines, puedes salir del entorno con `deactivate`.

---

## Paso 4 — Instalar dependencias

Son 5 paquetes estándar de Python científico:

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:

```
scikit-learn>=1.3,<2.0
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
```

Toma aproximadamente 1 minuto con buena conexión.

---

## Paso 5 — Ejecutar el experimento

Desde la raíz del proyecto:

```bash
python3 code/run_experiments.py
```

El script imprime el progreso en consola. Lo que verás:

```
======================================================================
1. CARGA Y PREPROCESAMIENTO
======================================================================
Dataset: 1998 registros, 22 columnas
Clases:
CLASE
1    1555
2     280
3     163

Train: 1698, Test: 300

======================================================================
2. ANÁLISIS EXPLORATORIO
======================================================================
Figuras EDA generadas

======================================================================
4. K-FOLD CV (5 folds) + GRID SEARCH
======================================================================

--- LogReg ---
Tiempo: 0.6s | Combinaciones: 8
Mejor F1-macro CV: 0.7890
Mejores params: {'clf__C': 10.0, 'clf__class_weight': None}

--- SVM ---
Tiempo: 3.1s | Combinaciones: 12
Mejor F1-macro CV: 0.8441
...

======================================================================
6. BOOTSTRAP .632 (500 réplicas)
======================================================================
LogReg: bootstrap... F1.632=0.8087 | OOB F1=0.7901 ...
SVM: bootstrap... F1.632=0.8910 | OOB F1=0.8399 ...
DecisionTree: bootstrap... F1.632=0.8968 | OOB F1=0.8515 ...
KNN: bootstrap... F1.632=0.8855 | OOB F1=0.8191 ...

======================================================================
7. MEJOR MODELO Y MATRIZ DE CONFUSIÓN EN TEST
======================================================================
Mejor modelo (por F1.632): DecisionTree
Accuracy en test: 0.9233
F1-macro en test: 0.8778
```

---

## Paso 6 — Verificar resultados

El script regenera estos archivos (sobrescribe los que vinieron en el ZIP):

```
results/grid_search_full.csv       64 combinaciones evaluadas
results/per_model_metrics.csv      Mejor config + métricas por clase
results/summary.json               Todo el experimento en JSON

figures/class_distribution.pdf     Distribución de las 3 clases
figures/correlation.pdf            Matriz de correlación
figures/model_comparison.pdf       Comparación F1-macro entre modelos
figures/confusion_matrix.pdf       Matriz de confusión del mejor modelo
```

**Verificación de reproducibilidad (semilla 42):**

Los números que aparecen en el paper deben coincidir *exactamente* con los
que aparecen en la consola al ejecutar:

| Resultado en el paper | Valor esperado |
|---|---|
| F1-macro CV Decision Tree | 0.882 |
| F1-macro bootstrap .632 Decision Tree | 0.897 |
| Accuracy en test | 92.33% |
| F1-macro en test | 0.878 |

Si algún número no coincide, lo más probable es que la versión de
scikit-learn sea muy diferente. El paper se generó con `scikit-learn==1.8.0`.

---

## Paso 7 — Recompilar el paper (opcional)

El PDF (`paper/main.pdf`) ya viene compilado. Si quieres regenerarlo desde
el `.tex`:

**Requisitos:** una distribución LaTeX que incluya `IEEEtran.cls`.

- **Ubuntu**: `sudo apt install texlive-latex-base texlive-publishers texlive-fonts-recommended`
- **macOS**: instala MacTeX desde https://tug.org/mactex/
- **Windows**: instala MiKTeX desde https://miktex.org

Luego:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex    # segunda pasada para resolver referencias cruzadas
```

Esto regenera `main.pdf` (5 páginas).

---

## Solución de problemas

### "ModuleNotFoundError: No module named 'sklearn'"
No activaste el entorno virtual o no instalaste los requisitos. Repite los
pasos 3 y 4.

### "FileNotFoundError: cardiotocographic.csv"
Estás ejecutando el script desde una carpeta equivocada. Debes estar en
`ctg_mini_proyecto/` (no dentro de `code/`).

```bash
cd /ruta/a/ctg_mini_proyecto
python3 code/run_experiments.py
```

### El script tarda mucho más que 5 minutos
Verifica que tu Python use todos los cores disponibles. El `GridSearchCV`
está configurado con `n_jobs=-1` (usa todos los CPUs). En máquinas con
1 core puede tardar hasta 15 minutos; sigue siendo correcto.

### Números ligeramente distintos a los del paper
La semilla `np.random.seed(42)` y `random_state=42` garantizan los mismos
resultados en la misma versión de sklearn. Si tu sklearn es muy distinto
(por ejemplo, 1.2 o anterior), pueden cambiar algunos decimales en los
métodos randomizados, pero las conclusiones del paper (Decision Tree gana,
Sospechoso es la clase más difícil) son robustas.

### "externally-managed-environment" al hacer pip install
Ubuntu 23.04+ y Debian 12+ bloquean pip global. Solución: usa el entorno
virtual del paso 3 (lo más limpio), o agrega `--break-system-packages` al
final del comando `pip install`.

---

## Resumen rápido (todo en uno)

Para alguien con Python ya instalado:

```bash
# 1. Descomprimir y entrar
cd ctg_mini_proyecto

# 2. Entorno + dependencias
python3 -m venv venv
source venv/bin/activate              # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# 3. Ejecutar
python3 code/run_experiments.py

# 4. Ver el paper
xdg-open paper/main.pdf               # Linux
# open paper/main.pdf                 # macOS
# start paper/main.pdf                # Windows
```

---

## Contacto

Si algo falla, revisa el `README.md` o contacta al equipo del proyecto.
