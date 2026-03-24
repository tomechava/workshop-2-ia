# Workshop 2 – Machine Learning & Deep Learning Aplicado

**Universidad EAFIT — Introducción a la Inteligencia Artificial (2026-01)**

---

## Descripción

Este repositorio contiene el desarrollo completo del Workshop 2, que integra dos problemas supervisados independientes: uno de **clasificación** y uno de **regresión**, aplicando el ciclo completo de un proyecto de Machine Learning y Deep Learning.

---

## Estructura del repositorio

```
workshop_2/
├── README.md                        ← Este archivo
├── requirements.txt                 ← Dependencias del proyecto
├── clasificacion/
│   ├── clasificacion.ipynb          ← Problema 1: Clasificación de fatiga muscular (EMG)
│   └── data/
│       └── nuevo_dataset.csv        ← Dataset generado por feature engineering
└── regresion/
    └── regresion.ipynb              ← Problema 2: Estimación de edad con CNN
```

---

## Problema 1 — Clasificación: Detección de Fatiga Muscular en Ciclismo

**Dataset:** [YominE/Muscle_Fatigue_Cycling](https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling) — HuggingFace

Señales EMG de 8 músculos de la pierna dominante durante sprints en bicicleta. El objetivo es clasificar el estado muscular del ciclista:
- `0` = Condición normal
- `1` = Desgaste muscular

**Pipeline:**
1. Carga del dataset desde HuggingFace y binarización del target
2. Feature engineering — ventanas de 1s, 7 características × 8 canales = 56 features
3. EDA completo con interpretaciones
4. Pipeline scikit-learn — normalización + split 70/15/15
5. Entrenamiento y comparación: kNN, Decision Tree, Random Forest, Gradient Boosting, DNN
6. Ajuste de hiperparámetros con Random Search
7. Evaluación final del mejor modelo (Random Forest — F1=0.82 en Test)
8. Prueba con muestra artificial

---

## Problema 2 — Regresión: Estimación de Edad a partir de Imágenes Faciales

**Dataset:** [arashnic/faces-age-detection-dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset) — Kaggle

Imágenes faciales etiquetadas con categorías de edad (YOUNG, MIDDLE, OLD). El objetivo es entrenar una CNN que estime la edad numérica del sujeto.

**Pipeline:**
1. Mapeo de categorías a edades numéricas (YOUNG→20, MIDDLE→45, OLD→70)
2. EDA — distribución de edades, balance de clases, muestras representativas
3. Pipeline con tf.data — redimensionamiento, normalización y data augmentation
4. CNN para regresión con salida lineal y loss MAE
5. Evaluación final: MAE=10.34 años, RMSE=13.10 años, R²=0.35
6. Prueba con imagen real del test set

---

## Cómo ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Abrir notebooks en VSCode o Jupyter
# Problema 1: clasificacion/clasificacion.ipynb
# Problema 2: regresion/regresion.ipynb
```

> **Nota Problema 1:** El dataset se descarga automáticamente desde HuggingFace al correr la primera celda. El archivo `data/nuevo_dataset.csv` se genera localmente tras el feature engineering.

> **Nota Problema 2:** El dataset debe descargarse manualmente desde Kaggle y la ruta `BASE_PATH` en el notebook debe actualizarse a la ubicación local de las imágenes.

---

## Equipo

| Integrante | Problema |
|---|---|
| Tomas Echavarria | Clasificación — Problema 1 |
| Nathan Martinez | Regresión — Problema 2 |

**Universidad EAFIT — 2026-01**