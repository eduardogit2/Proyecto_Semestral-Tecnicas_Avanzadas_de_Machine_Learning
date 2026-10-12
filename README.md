# Clasificación de razas de gatos con redes neuronales

**TLY1102 Técnicas de Machine Learning I** (Duoc UC) · **Grupo 7:** Agustín Fernández · Sebastián Salgado · Eduardo Uribe

## Descripción

Este proyecto aborda un problema de clasificación de imágenes: a partir de una fotografía, identificar la raza de un gato. Se trabaja con 12 razas del Oxford-IIIT Pet Dataset, y se utilizan distintos tipos de redes neuronales para resolverlo y compararlos.

El problema es de grano fino, porque varias razas comparten color, tamaño y rasgos faciales, así que las diferencias relevantes están en detalles de textura y forma.

## Enfoque

El proyecto explora cómo distintas arquitecturas de redes neuronales enfrentan el mismo problema:

- **Perceptrón Multicapa (MLP):** trabaja con la imagen aplanada en un vector, lo que elimina la relación espacial entre píxeles.
- **Redes Neuronales Convolucionales (CNN):** conservan la estructura de la imagen y aprenden filtros locales que extraen características jerárquicas, desde bordes hasta partes del animal.

Para cada modelo se revisan las métricas de clasificación, el comportamiento durante el entrenamiento, los errores y la interpretabilidad, mediante Feature Maps, etc.

## Datos

- Imágenes de gatos de 12 razas, redimensionadas a 64 × 64 píxeles en RGB y normalizadas al rango [0, 1].
- Partición en entrenamiento, validación y prueba, con la misma semilla en todo el proyecto para que los resultados sean comparables.
- El conjunto de datos se descarga automáticamente al ejecutar los notebooks.

## Estructura del repositorio

```
.
├── README.md
├── data/        # Dataset (se descarga al ejecutar los notebooks)
├── notebooks/   # Notebooks con el desarrollo
├── models/      # Modelos entrenados
├── images/      # Figuras y resultados
└── docs/        # Informes técnicos
```

## Requisitos

Python con las siguientes librerías (los notebooks se desarrollaron con TensorFlow 2.21.0):

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Ejecución

Los notebooks deben ejecutarse desde la carpeta `notebooks/`, porque las rutas son relativas. La primera ejecución requiere conexión a internet, ya que el dataset y algunos modelos se descargan automáticamente si no existen localmente.

Las semillas están fijadas en 29, por lo que los resultados son reproducibles con la misma versión de TensorFlow.


## Matriz de responsabilidades

| Integrante | Tareas realizadas |
|---|---|
| **Agustín Fernández** | Preparación de datos, lógica de extracción del dataset, validación de normalización y EDA |
| **Sebastián Salgado** | Diseño e implementación de la arquitectura CNN, visualización de Feature Maps |
| **Eduardo Uribe** | Entrenamiento, evaluación comparativa CNN vs MLP (E1), análisis Grad-CAM y redacción final |
