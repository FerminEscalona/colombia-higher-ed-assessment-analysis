# Figuras para informes

Estas imágenes PNG se generan con el mismo archivo de resultados que utiliza el
dashboard. Sirven como insumos estáticos para informes, presentaciones o documentos
académicos.

| Archivo | Uso sugerido |
|---|---|
| `01_distribucion_perfiles_kmeans.png` | Tamaño relativo de cada perfil K-Means. |
| `02_resiliencia_por_perfil.png` | Concentración de resiliencia y promedio de la población. |
| `03_proyeccion_pca_kmeans.png` | Separación visual de los perfiles K-Means en dos componentes. |
| `04_perfiles_contextuales_estandarizados.png` | Comparación relativa de puntaje, INSE, bienes y resiliencia. |
| `05_bienes_y_servicios_por_perfil.png` | Disponibilidad de bienes del hogar por perfil. |
| `06_educacion_materna_y_resiliencia.png` | Comparación descriptiva entre educación materna y resiliencia. |
| `07_distribucion_dbscan.png` | Tamaño de grupos y ruido identificados por DBSCAN. |
| `08_proyeccion_pca_dbscan.png` | Estructura visual de DBSCAN en dos componentes. |
| `09_metricas_internas_modelos.png` | Tabla de métricas internas de ambos modelos. |

Para regenerarlas después de entrenar los modelos:

```bash
python -m src.visualization.export_figures
```

Las figuras son descriptivas. No deben utilizarse para afirmar causalidad ni para
clasificar individualmente a un estudiante o institución.
