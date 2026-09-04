# Perfiles de resiliencia académica en Saber Pro 2024

Proyecto de ciencia de datos sobre los resultados de las competencias genéricas del Examen de Estado de la Calidad de la Educación Superior **Saber Pro 2024** en Colombia. Sigue las fases de CRISP-DM: entendimiento del negocio y de los datos, preparación, modelado y evaluación.

La pregunta central del estudio es:

> ¿Qué caracteriza a los perfiles de resiliencia académica: estudiantes con bajo índice socioeconómico (INSE) que logran ubicarse en percentiles sobresalientes en la prueba?

El proyecto es exploratorio. Los resultados describen patrones observados; no demuestran que una condición familiar, institucional o territorial cause un resultado académico.

## Alcance

La unidad de análisis es un registro individual de Saber Pro. La fuente reúne información del estudiante, su hogar, programa académico, institución y resultados en Lectura Crítica, Razonamiento Cuantitativo, Competencias Ciudadanas, Comunicación Escrita e Inglés.

Para el análisis de resiliencia se consideran estudiantes institucionales con contexto socioeconómico disponible y resultados válidos. La población modelada corresponde al cuartil inferior del INSE. Dentro de ella, se considera resiliente a quien alcanza un percentil global igual o superior a 90.

## Qué abordamos

1. **Entendimiento del negocio:** contexto de Saber Pro, actores, variables, riesgos de interpretación y límites del archivo.
2. **EDA:** tipos de datos, distribución, faltantes, posibles atípicos, categorías, correlaciones y relaciones descriptivas.
3. **Preparación de datos:** tratamiento documentado de faltantes según su contexto: inscripción individual, cuestionario exterior, campo libre no diligenciado o resultado no válido. El archivo original no se modifica.
4. **Clustering:** perfiles con K-Means y una lectura complementaria con DBSCAN de grupos densos y casos poco comunes.
5. **Evaluación y comunicación:** métricas internas, perfiles, concentración de resiliencia y un dashboard interactivo en Streamlit.

### Variables de los modelos

Los modelos usan variables de contexto: estrato, educación y ocupación parental, bienes del hogar y acciones reportadas de preparación para el examen. Las variables ordinales se estandarizan; las binarias y categorías codificadas se mantienen en escala 0/1. Las categorías muy poco frecuentes se agrupan como `OTRAS CATEGORÍAS`.

El puntaje global, percentil global, INSE y la etiqueta de resiliencia se reservan para interpretar los perfiles: no se usan para crearlos.

### Modelos y resultados

- **K-Means:** construye cuatro perfiles para la población de bajo INSE: mayor vulnerabilidad observada, mayor disponibilidad material, preparación académica reportada y mayor capital educativo familiar.
- **DBSCAN:** identifica seis grupos en una muestra de 30.000 estudiantes y marca como ruido los casos fuera de grupos densos. Se usa solo para análisis descriptivo; no asigna nuevos estudiantes.
- **Evaluación:** se calculan Silhouette, Davies-Bouldin y Calinski-Harabasz. Son métricas de estructura interna, no métricas de predicción ni de causalidad.

Con la ejecución reproducible actual se modelan 61.943 estudiantes de bajo INSE; 2.208 (3,56 %) cumplen la definición operativa de resiliencia. Los valores pueden cambiar al utilizar una versión diferente de la fuente.

## Datos

El archivo de entrada no se publica en el repositorio. Debe ubicarse localmente en:

```text
data/raw/Examen_Saber_Pro_Genericas_2024.txt
```

El notebook de preparación lo lee con separador `;` y codificación `utf-8-sig`. Genera una copia limpia sin reemplazar el original:

```text
data/processed/Examen_Saber_Pro_Genericas_2024_limpio.csv
```

Los directorios `data/` y `models/` están ignorados por Git. Cada persona debe disponer del archivo fuente autorizado y regenerar los resultados localmente.

## Estructura

```text
.
├── config/                         # Parámetros de datos y modelado
├── data/
│   ├── raw/                        # Fuente original, no versionada
│   └── processed/                  # Datos limpios y resultados de clustering
├── notebooks/
│   ├── 1.0-business-understanding.ipynb
│   ├── 2.0-data-understanding.ipynb
│   ├── 3.0-data-prep.ipynb
│   ├── 4.0-modeling.ipynb
│   └── 5.0-evaluation.ipynb
├── src/
│   ├── data/                       # Rutas y carga de datasets
│   ├── features/                   # Transformación reproducible de variables
│   ├── models/                     # Entrenamiento, evaluación y predicción
│   ├── visualization/              # Gráficas reutilizables de Plotly
│   └── dashboard/                  # Aplicación Streamlit
│       ├── app.py                  # Punto de entrada
│       └── views/                  # Vistas internas
├── models/                         # Artefactos locales serializados
├── tests/                          # Pruebas de reproducibilidad y arranque
├── requirements.txt
└── README.md
```

## Cómo ejecutar el proyecto

### 1. Crear el entorno

Se requiere Python 3.10 o superior.

```bash
git clone <URL_DEL_REPOSITORIO>
cd colombia-higher-ed-assessment-analysis

python -m venv .venv
source .venv/bin/activate

# En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Ubicar la fuente

Copie el archivo autorizado de Saber Pro 2024 en `data/raw/Examen_Saber_Pro_Genericas_2024.txt`.

### 3. Ejecutar el análisis

Puede abrir los notebooks con la extensión de Jupyter de VS Code y seleccionar el
entorno `.venv` como kernel. Para usar JupyterLab desde terminal, instálelo una vez
en el entorno e inícielo desde la raíz del proyecto:

```bash
pip install jupyterlab
jupyter lab
```

Ejecute los notebooks en este orden:

1. `1.0-business-understanding.ipynb`: contexto, variables y límites.
2. `2.0-data-understanding.ipynb`: exploración descriptiva y calidad de datos.
3. `3.0-data-prep.ipynb`: tratamiento de faltantes y creación del CSV limpio.
4. `4.0-modeling.ipynb`: entrenamiento reproducible de K-Means y DBSCAN.
5. `5.0-evaluation.ipynb`: métricas, perfiles y conclusiones.

Los dos primeros notebooks no modifican datos. El tercero produce el CSV necesario para modelado y evaluación.

### 4. Entrenar desde terminal (opcional)

Después de generar el CSV limpio, se puede reproducir el modelado sin abrir Jupyter:

```bash
python -m src.models.train_model
```

Esto crea localmente:

```text
data/processed/saber_pro_2024_clusters_estudiantes.csv
models/saber_pro_clustering_bundle.joblib
models/model_summary.json
```

El archivo `.joblib` conserva transformaciones, PCA y K-Means entrenados. El JSON resume parámetros, métricas y nombres de perfiles, para mantener el dashboard trazable y reproducible.

### 5. Iniciar el dashboard

```bash
streamlit run src/dashboard/app.py
```

Abra la dirección indicada por Streamlit, normalmente `http://localhost:8501`. La aplicación incluye filtros por perfil, departamento, metodología, origen institucional y núcleo académico, además de vistas de resumen, perfiles, resiliencia, DBSCAN y metodología.

### 6. Exportar figuras para informes

Después del modelado, genere las figuras PNG estáticas usadas en informes y presentaciones:

```bash
python -m src.visualization.export_figures
```

Las imágenes se guardan en `reports/figures/`. Consulte el
[`README` de figuras](reports/figures/README.md) para conocer el propósito de cada una.

## Pruebas

Para verificar el pipeline de predicción, la reproducibilidad de K-Means y el arranque de las vistas del dashboard:

```bash
python -m unittest discover -v
```

## Uso responsable

- `estu_consecutivo`, códigos institucionales y códigos SNIES son identificadores; no son variables numéricas explicativas.
- Un percentil expresa posición relativa, no porcentaje de respuestas correctas.
- `-1` en el dataset limpio es un centinela contextual para un resultado no válido o no aplicable; no equivale a un puntaje real.
- La ausencia de reporte étnico se trató bajo un supuesto documentado en el notebook de preparación y debe interpretarse con cautela.
- Los perfiles no son diagnósticos individuales, rankings institucionales ni una prueba de causalidad. Se debe preservar la privacidad y evitar conclusiones estigmatizantes.

## Tecnologías

Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, Matplotlib, Seaborn y Jupyter.
