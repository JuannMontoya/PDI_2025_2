# 🎱 BilliardSync: Detección y Análisis de Billar (End-to-End)

![YOLO11](https://img.shields.io/badge/YOLO-v11-blue?style=for-the-badge&logo=ultralytics)
![LiteRT](https://img.shields.io/badge/LiteRT-TFLite-orange?style=for-the-badge&logo=tensorflow)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle)
![Hugging Face](https://img.shields.io/badge/Deployment-Spaces-yellow?style=for-the-badge&logo=huggingface)

## 📖 Descripción del Proyecto

Es un sistema de visión artificial de alto rendimiento desarrollado íntegramente en la nube (Kaggle) para la detección precisa de elementos en una mesa de billar. El proyecto abarca un flujo MLOps completo: desde el entrenamiento personalizado de una arquitectura YOLO11 Nano, pasando por la optimización del modelo a formato LiteRT (TensorFlow Lite) para inferencia de baja latencia, hasta su despliegue productivo en Hugging Face Spaces.

El objetivo es identificar no solo las bolas de juego, sino también la geometría de la mesa (troneras, intersecciones y bordes) para permitir futuras aplicaciones de realidad aumentada o arbitraje automático.

---

## ⚙️ Arquitectura y Entrenamiento (Kaggle Pipeline)

El núcleo del proyecto se basa en **YOLO11n**, seleccionado por su eficiencia (velocidad/precisión). El entrenamiento se ejecutó en aceleradores GPU P100/T4 de Kaggle con la siguiente configuración estratégica:

### 1. Configuración de Hiperparámetros
Para maximizar la precisión en objetos pequeños (bolas) y características geométricas:
* **Modelo:** `yolo11n.pt` (Preentrenado en COCO).
* **Resolución:** 640x640 píxeles.
* **Épocas:** 200 (con **Early Stopping** activado, `patience=20` y `min_delta=0.001`).
* **Batch Size:** 64 (Optimizado para VRAM de Kaggle).
* **Optimizador:** Auto (ajuste dinámico de Learning Rate).

### 2. Aumento de Datos (Data Augmentation)
Se aplicó una estrategia agresiva para combatir el sobreajuste y mejorar la generalización:
* **Mosaic (100%):** Fundamental para detectar objetos en contextos complejos.
* **Geometría:** Rotaciones (+/- 10°), Escala (+/- 50%) y Volteo Horizontal.
* **MixUp:** Desactivado para mantener la integridad de las formas geométricas precisas.

### 3. Ajuste de Funciones de Pérdida (Loss Tuning)
* `box=7.5`: Se priorizó drásticamente la regresión de cajas para asegurar que las bounding boxes se ajusten perfectamente a las bolas y troneras.
* `dfl=1.5`: Focal Loss para refinar la distribución.
* `cls=0.5`: Peso estándar para la clasificación de clases.

### 4. Optimización a LiteRT
Tras el entrenamiento, el modelo `.pt` fue convertido a **TensorFlow Lite (LiteRT)** usando la API de exportación de Ultralytics. Esto permite que el modelo sea agnóstico a la plataforma y corra eficientemente en CPUs o dispositivos móviles sin dependencias de PyTorch.

---

## 📊 Análisis de Resultados y Métricas

El modelo ha demostrado un rendimiento excepcional, especialmente en la detección de la estructura de la mesa, con una velocidad de inferencia extremadamente rápida.

### ⚡ Rendimiento de Inferencia (Benchmark)
* **Preproceso:** 1.0ms
* **Inferencia:** **3.8ms** (Tiempo real estricto, >200 FPS potenciales)
* **Postproceso:** 1.9ms

### 📈 Métricas Globales (Validación)
| Métrica | Valor Final | Interpretación |
| :--- | :--- | :--- |
| **mAP@50** | **0.836** | Alta fiabilidad en la detección general (IoU 0.5). |
| **mAP@50-95** | **0.552** | Excelente precisión de ajuste de caja (riguroso). |
| **Precision** | **0.791** | Baja tasa de falsos positivos. |
| **Recall** | **0.801** | El modelo encuentra el 80% de los objetos presentes. |

### 🎯 Desglose por Clases (Insights)

**1. Geometría de la Mesa (Rendimiento Perfecto):**
Las esquinas y puntos estructurales presentan una detección casi infalible.
* `BottomLeft`, `TopRight`, `MediumLeft`: **mAP@50 > 0.99**
* Esto garantiza que el sistema entiende perfectamente los límites del área de juego.

**2. Bolas de Billar (Rendimiento Sólido):**
Las bolas numeradas muestran un rendimiento consistente, con algunas variaciones debidas probablemente a oclusiones o reflejos.
* **Mejores:** Bola 0 (Blanca) y Bola 1 (~0.85 - 0.90 mAP).
* **Promedio:** La mayoría de las bolas oscilan entre **0.75 y 0.80 mAP**.
* **Desafíos:** La Bola 4 y 10 presentan métricas ligeramente inferiores (~0.70 mAP), candidatos para mejora con más datos de entrenamiento específicos.

---

## 💻 Ejecución del Proyecto (Kaggle)

Este proyecto no requiere instalación local compleja. Todo el entorno reside en Kaggle.

1.  **Abrir Notebook:** Accede al notebook 
2.  **Dataset:** Asegúrate de que el dataset esté conectado en el directorio `/kaggle/input`.
3.  **Ejecutar Entrenamiento:**
    ```python
    model.train(data='/kaggle/working/data.yaml', epochs=200, imgsz=640, ...)
    ```
4.  **Generar Reportes:** El código genera automáticamente gráficos de curvas de pérdida y matrices de confusión en `runs/detect/train`.

---

## 🌐 Despliegue (Hugging Face Spaces)

El modelo final optimizado se encuentra desplegado para pruebas públicas.

* **Framework:** Gradio SDK.
* **Modelo en uso:** Versión `best.pt` (o su variante LiteRT según configuración).
* **Funcionalidad:** Sube una imagen o video de una partida de billar y obtén las detecciones renderizadas al instante.

🔗 **[Probar Demo en Hugging Face](https://huggingface.co/spaces/JuannMontoya/billar-detector-v1)** *(Enlace al Space)*

---

## 📝 Créditos

Desarrollado con **Ultralytics YOLO11** y **Kaggle Kernels**.

MIT License.