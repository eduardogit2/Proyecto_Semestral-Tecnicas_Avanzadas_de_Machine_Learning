# Clasificación de Razas de Gatos mediante Perceptrón Multicapa (MLP)
Grupo 7: Agustín Fernández, Sebastián Salgado y Eduardo Uribe
Asignatura: Técnicas Avanzadas de Machine Learning I (TLY1101)
Evaluación: Parcial N.º 1

Descripción del proyecto
Este proyecto desarrolla un modelo de Machine Learning basado en un Perceptrón Multicapa (MLP) para clasificar automáticamente imágenes de gatos según su raza.

La problemática se relaciona con el sector veterinario y de adopción de mascotas, donde identificar correctamente una raza puede ser útil para considerar cuidados específicos, alimentación y posibles predisposiciones genéticas. Actualmente, este proceso puede depender de la experiencia visual del personal, lo que puede producir errores o demoras.

Para el desarrollo se utiliza como fuente el Oxford-IIIT Pet Dataset.
Variante utilizada
El dataset original contiene 37 clases de perros y gatos. Para este proyecto se seleccionó exclusivamente el subconjunto correspondiente a 12 razas de gatos, permitiendo especializar el análisis en clasificación de imágenes felinas.

Objetivos
Implementar y entrenar un modelo MLP para clasificar imágenes pertenecientes a 12 razas de gatos.
Comprender y documentar el flujo completo de un proyecto de aprendizaje supervisado:
Carga de datos.
Preprocesamiento.
Modelamiento.
Entrenamiento.
Validación.
Evaluación.
Analizar críticamente el desempeño y las limitaciones de un MLP aplicado a imágenes.

Resultados y KPIs

El modelo no alcanzó la meta de Accuracy propuesta.

Durante el entrenamiento se observó que, después de 30 épocas, el modelo todavía no había convergido. Los resultados aproximados fueron:

Accuracy de entrenamiento: 47%
Accuracy de validación: 46%
Accuracy de test: 25%

Esto indica principalmente un problema de bajo aprendizaje (underfitting), ya que las curvas de entrenamiento continuaban aumentando al finalizar las épocas y no se observó un sobreajuste marcado.

También se identificó una diferencia importante entre validación y test, aproximadamente 46% frente a 25%, la cual constituye uno de los principales hallazgos para analizar durante la defensa.

Fuente de datos
Oxford-IIIT Pet Dataset
Fuente: Oxford-IIIT Pet Dataset.
Contenido utilizado: subconjunto de 12 razas de gatos.
Filtrado: a partir de las anotaciones oficiales trainval.txt y test.txt, utilizando SPECIES == 1.
Imágenes de entrenamiento inicial: aproximadamente 1.896.
Imágenes de test: aproximadamente 475.
Distribución: aproximadamente 147–160 imágenes por raza en entrenamiento y 37–40 en test.
Formato: imágenes RGB.
Tamaño utilizado por el modelo: 64 × 64 píxeles.

Análisis exploratorio de datos (EDA)
Durante el análisis exploratorio se realizaron las siguientes tareas:

Verificación de la distribución de clases en los conjuntos de entrenamiento y test.
Inspección de las dimensiones originales de las imágenes.
Visualización de ejemplos pertenecientes a cada raza.
Identificación de posibles dificultades de clasificación.

La distribución de las clases se encuentra aproximadamente balanceada, sin observarse un desbalance severo.

Entre las principales dificultades visuales se identificaron:

Variaciones de iluminación.
Diferencias de fondo.
Distintas posiciones de los gatos.
Variaciones de escala.
Similitudes visuales entre determinadas razas.
Preprocesamiento
Las imágenes fueron procesadas mediante:

Carga utilizando image_dataset_from_directory.
Redimensionamiento a 64 × 64 píxeles.
Normalización de los valores de píxel al rango [0, 1].
División de los datos en:
Train: 80%.
Validation: 20%.
Test: conjunto de prueba.

Metodología CRISP-DM
El proyecto sigue las principales fases de la metodología CRISP-DM:


Arquitectura del modelo
El modelo utilizado corresponde a un Perceptrón Multicapa (MLP):

Flujo simplificado
Imagen RGB 64×64×3

        ↓

     Flatten

        ↓

   12.288 valores

        ↓

Capa densa: 128 + ReLU

        ↓

Capa densa: 64 + ReLU

        ↓

Salida: 12 + Softmax

        ↓

Raza predicha

Limitaciones identificadas
El modelo presenta varias limitaciones importantes:
1. Pérdida de información espacial
La utilización de Flatten transforma la imagen en un vector, eliminando las relaciones espaciales entre los píxeles.
2. Alta dimensionalidad
Incluso utilizando imágenes pequeñas de 64 × 64 píxeles, la entrada contiene 12288 valores.
3. Sensibilidad a las características de la imagen
El modelo puede verse afectado por:

Posición del gato.
Iluminación.
Fondo.
Escala.
Características visuales compartidas entre razas.
4. Arquitectura no especializada para visión
Un MLP no está diseñado específicamente para trabajar con imágenes. Una arquitectura CNN (Convolutional Neural Network) permitiría aprovechar mejor la estructura espacial de las imágenes.
5. Bajo aprendizaje
Con la configuración utilizada —SGD, learning rate = 0.01 y 30 épocas— el modelo no logró converger completamente.
6. Diferencia entre validación y test
Se observó una brecha relevante entre el rendimiento de validación y test:

Validation ≈ 46% vs. Test ≈ 25%

Esta diferencia debe ser considerada dentro del análisis del proyecto y de la defensa.
7. Confusiones frecuentes
Las principales confusiones observadas fueron:

Abyssinian ↔ Bengal
British Shorthair ↔ Russian Blue / Sphynx
Persian / Sphynx → Bengal

Estas confusiones pueden explicarse por similitudes en características como color, textura del pelaje y apariencia general.

Consideraciones éticas
El dataset utilizado no contiene datos personales ni sensibles, ya que corresponde a imágenes de animales.
El modelo debe utilizarse como herramienta de apoyo, no como reemplazo del criterio de un profesional veterinario.
Una clasificación incorrecta podría llevar a decisiones de cuidado inadecuadas si se utiliza sin supervisión humana.
La representación del dataset es limitada, ya que solo se consideran 12 razas y un número reducido de imágenes por clase.
Estas limitaciones pueden generar resultados menos representativos frente a razas o características no presentes en los datos.

Estructura del proyecto
proyecto-mlp-gatos/

│

├── README.md

│

├── notebooks/

│   └── TLY1102_E1_Desarrollo_CORREGIDO_v2.ipynb

│

├── data/

│   └── (imágenes del Oxford-IIIT Pet Dataset)

│

├── models/

│   └── (modelo entrenado)

│

└── images/

    └── (gráficos exportados: curvas de entrenamiento,

        matriz de confusión, etc.)

Notebook
El desarrollo y experimentación del modelo se encuentra en:

notebooks/TLY1102_E1_Desarrollo_CORREGIDO_v2.ipynb

En el notebook se documentan las etapas de carga de datos, preparación, entrenamiento, validación y evaluación del modelo.

Mejoras futuras
Como líneas de trabajo futuro se propone:

Probar diferentes hiperparámetros.
Evaluar otros optimizadores y tasas de aprendizaje.
Aumentar el número de épocas cuando sea apropiado.
Aplicar técnicas de Data Augmentation.
Comparar el MLP con arquitecturas CNN.
Analizar con mayor profundidad la brecha entre validación y test.
Evaluar otras métricas y estrategias para mejorar la clasificación de las clases con mayor confusión.
Incorporar una etapa de despliegue del modelo.

Conclusiones
El proyecto permitió implementar un flujo completo de clasificación supervisada utilizando un Perceptrón Multicapa (MLP) sobre una variante del Oxford-IIIT Pet Dataset, enfocada exclusivamente en 12 razas de gatos.

El preprocesamiento permitió homogeneizar las imágenes y normalizar sus valores de píxel. La arquitectura utilizada cuenta con dos capas ocultas con activación ReLU y una capa de salida Softmax de 12 clases.

Aunque el modelo no alcanzó la meta de Accuracy establecida, los resultados permitieron identificar claramente las limitaciones de utilizar un MLP para clasificación de imágenes. En particular, el uso de Flatten provoca la pérdida de información espacial y dificulta la identificación de características visuales relevantes.

Como principal mejora futura se plantea comparar diferentes configuraciones de hiperparámetros y evaluar arquitecturas CNN, las cuales están mejor adaptadas al procesamiento de imágenes.

Integrantes
Agustín Fernández
Sebastián Salgado
Eduardo Uribe

Técnicas Avanzadas de Machine Learning I — TLY1101
Evaluación Parcial N.º 1

KPI | Meta | Resultado obtenido | Cumplimiento
Accuracy (Test) | > 60% | 24,6% | No cumplido
F1-Score (Macro) | Equilibrado entre clases | 23,2% | Parcial
Precision | — | 26,4% | —
Recall | — | 24,8% | —
Fase | Aplicación en el proyecto
Comprensión del negocio | Definición del problema, objetivos y KPIs.
Comprensión de los datos | EDA, distribución de clases, tamaños y ejemplos de imágenes.
Preparación de los datos | Redimensionamiento, normalización y partición Train/Validation/Test.
Modelamiento | Implementación de un Perceptrón Multicapa con dos capas ocultas.
Evaluación | Accuracy, Precision, Recall, F1-Score, matriz de confusión y análisis de errores.
Despliegue | No abordado en esta primera experiencia; queda como trabajo futuro.
Componente | Configuración
Entrada | 64 × 64 × 3
Flatten | 12288  valores
Capa oculta 1 | 128 neuronas + ReLU
Capa oculta 2 | 64 neuronas + ReLU
Salida | 12 neuronas + Softmax
Función de pérdida | Categorical Crossentropy
Optimizador | SGD
Learning rate | 0.01
Batch size | 32
Épocas máximas | 30
EarlyStopping | Paciencia de 5 sobre val_loss
Semilla | 29