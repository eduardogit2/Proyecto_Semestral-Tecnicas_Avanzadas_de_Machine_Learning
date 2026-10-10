# Proyecto Semestral – Técnicas Avanzadas de Machine Learning

## 1. Matriz de Responsabilidades
| Integrante | Rol / Responsabilidad en el Proyecto |
| :--- | :--- |
| **Agustín Fernández** | Preparación de datos (Data Pipeline), entrenamiento y validación de modelos. |
| **Sebastián Salgado** | Análisis Exploratorio de Datos (EDA), interpretación de curvas y Feature Maps. |
| **Eduardo Uribe** | Diseño de arquitectura de red neuronal, evaluación de métricas y documentación. |

## 2. Descripción del Problema y del Negocio
El problema consiste en construir un modelo de Machine Learning capaz de clasificar automáticamente imágenes de gatos según su raza. El modelo se crea como una herramienta de apoyo a la identificación visual, no como un reemplazo del criterio profesional veterinario. 

**Variante del Proyecto:** Se filtró el dataset original (Oxford-IIIT Pet Dataset) aislando la variable `SPECIES == 1` para resolver el problema de clasificación exclusivamente para las 12 razas de gatos disponibles.

## 3. Objetivos y KPIs
*   **Objetivo Principal:** Implementar y entrenar un Perceptrón Multicapa (MLP) y una Red Neuronal Convolucional (CNN) para clasificar 12 razas de gatos.
*   **KPI 1 (Accuracy):** Lograr una precisión global en el conjunto de prueba (Test) superior al 60%.
*   **KPI 2 (F1-Score):** Obtener un F1-Score equilibrado para asegurar el correcto reconocimiento de las razas minoritarias.

## 4. Metodología Utilizada (CRISP-DM)
El proyecto se rige bajo la metodología CRISP-DM:
1.  **Comprensión del Negocio:** Definición de objetivos enfocados en la clasificación visual felina.
2.  **Comprensión de los Datos:** Extracción y análisis de las imágenes.
3.  **Preparación de los Datos:** Redimensionamiento de imágenes, normalización de píxeles al rango [0.0, 1.0] y separación de sets.
4.  **Modelamiento:** Creación de arquitectura Perceptrón Multicapa (MLP) base y posterior rediseño a Red Neuronal Convolucional (CNN).
5.  **Evaluación:** Análisis de sobreajuste, matrices de confusión y métricas.
6.  **Despliegue:** Documentación estructurada y reproducible en repositorio.

## 5. Descripción de las Fuentes de Datos y Análisis Exploratorio (EDA)
Se utilizó el **Oxford-IIIT Pet Dataset**. Debido al tamaño del dataset (~1GB), las imágenes no están incluidas en este repositorio. Para reproducir este proyecto, ejecuta las celdas de descarga de datos incluidas en el Notebook.
*   **Volumen:** 1.896 imágenes de entrenamiento y 475 de prueba.
*   **Clases (12):** Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx.
*   **Distribución:** El dataset se encuentra balanceado, con un promedio de 150 imágenes por raza para entrenamiento y 40 para testing.
*   **Partición:** 80% Entrenamiento / 20% Prueba.

## 6. Justificación de las Decisiones Adoptadas (Arquitectura)
*   **MLP (Evaluación 1):** Se utilizó una capa `Flatten` seguida de capas densas (`Dense`) con activación `ReLU` para extracción de características lineales, finalizando en una capa de salida `Softmax` de 10-12 nodos para la clasificación probabilística. Optimizador: Adam. Función de pérdida: `sparse_categorical_crossentropy`.
*   **CNN (Evaluación 2):** *(Sección a completar durante el desarrollo)*.

## 7. Conclusiones y Análisis de Resultados
*(Esta sección se completará tras contrastar el rendimiento del MLP de la Evaluación 1 contra la arquitectura CNN de la Evaluación 2).*