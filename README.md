# Clasificación de razas de gatos con redes neuronales

**TLY1102 – Técnicas de Machine Learning I (Duoc UC)**  
**Grupo 7:** Agustín Fernández · Sebastián Salgado · Eduardo Uribe

## Descripción

Este proyecto busca identificar la raza de un gato a partir de una imagen. Para ello, trabajamos con 12 razas del **Oxford-IIIT Pet Dataset** y desarrollamos distintos modelos de redes neuronales.

Se trata de un problema de clasificación de grano fino, ya que algunas razas presentan características visuales similares, como el color del pelaje, la forma del rostro y la textura.

## Desarrollo del proyecto

A lo largo de las evaluaciones del curso, exploramos y comparamos dos arquitecturas:

- **Perceptrón Multicapa (MLP):** procesa las imágenes como vectores de píxeles.
- **Red Neuronal Convolucional (CNN):** aprovecha la estructura espacial de las imágenes para aprender características visuales.

El trabajo incluye preparación y exploración de datos, entrenamiento, evaluación, regularización y optimización de los modelos. También analizamos sus errores y utilizamos herramientas de interpretabilidad, como mapas de características, Grad-CAM y mapas de saliencia.

Los experimentos, resultados, gráficos y conclusiones de cada etapa se encuentran documentados en los notebooks y en los informes del repositorio.

## Datos

- **Origen:** Oxford-IIIT Pet Dataset, utilizando las imágenes de 12 razas de gatos.
- **Formato de entrada:** imágenes RGB de 64 × 64 píxeles.
- **Preprocesamiento:** normalización de los valores de píxeles al rango [0, 1].
- **Evaluación:** particiones de entrenamiento, validación y prueba.

El conjunto de datos se descarga automáticamente cuando se ejecutan los notebooks que lo requieren, siempre que exista conexión a internet.

## Estructura del repositorio

```text
.
├── README.md
├── data/          # Dataset y archivos de datos
├── docs/          # Informes y documentación
├── images/        # Gráficos y visualizaciones
├── models/        # Modelos entrenados
├── notebooks/     # Desarrollo de las evaluaciones
└── reportes/      # Tablas y resultados exportados
```

Algunas carpetas se crean automáticamente durante la ejecución, si aún no existen.

## Requisitos

El proyecto utiliza **Python**, **TensorFlow/Keras** y librerías para el análisis y la visualización de datos.

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Ejecución

1. Clona o descarga el repositorio.
2. Abre el proyecto en VS Code o Jupyter Notebook.
3. Abre el notebook que quieras revisar dentro de `notebooks/`.
4. Ejecuta sus celdas en orden y sigue las indicaciones de cada evaluación.

Los notebooks más recientes están configurados para utilizar las carpetas principales del proyecto al ejecutarse desde la raíz del repositorio o desde `notebooks/`.

El entrenamiento puede tardar según el equipo utilizado. Las semillas ayudan a reproducir los experimentos, pero algunos resultados pueden variar ligeramente entre entornos.

## Matriz de responsabilidades

| Integrante | Tareas realizadas |
|---|---|
| **Agustín Fernández** | Preparación de datos, extracción del dataset, validación de normalización y análisis exploratorio. |
| **Sebastián Salgado** | Diseño e implementación de la CNN y visualización de mapas de características. |
| **Eduardo Uribe** | Entrenamiento, evaluación comparativa de MLP y CNN, análisis Grad-CAM y documentación final. |
