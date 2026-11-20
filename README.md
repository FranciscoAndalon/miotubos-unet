# Segmentación inicial de miotubos

Implementación local de segmentación **semántica binaria**: miotubo frente a fondo. Incluye una línea base clásica y una U-Net pequeña escrita directamente en PyTorch, sin pesos preentrenados ni plataformas de entrenamiento automatizado. Las imágenes y las etiquetas originales se leen sin modificarlas.

## Resultado de la ejecución incluida

Entrenamiento local en Apple M1, 8 GB de RAM, GPU Metal/MPS. U-Net tiene **275,161 parámetros**. Se ejecutaron 32 épocas en aproximadamente 10.3 minutos de entrenamiento y validación, con parada temprana; se guardó la época 22, con umbral 0.5. El checkpoint ocupa aproximadamente 1.1 MB.

Resultados en **46 imágenes de prueba**, después de devolver las predicciones a **2592×1944**. Métricas micro, calculadas acumulando píxeles de todo el conjunto:

| Método | Dice | IoU | Precisión | Sensibilidad |
|---|---:|---:|---:|---:|
| Línea base clásica | 0.1910 | 0.1056 | 0.1207 | 0.4566 |
| U-Net pequeña | **0.3902** | **0.2424** | **0.2611** | **0.7720** |

Dice macro en imágenes con primer plano real: **0.3746** para U-Net y **0.2039** para la línea base. Dice macro incluyendo todas las imágenes: 0.3502 y 0.1906, respectivamente. Ambos métodos producen algún falso positivo en las tres imágenes de prueba con máscara vacía. En validación, el Dice macro no vacío fue 0.2322 para el checkpoint y 0.1101 para la línea base; la diferencia con prueba refleja que los grupos tienen dificultades distintas, no una selección posterior sobre prueba.

La mejora de la red es clara frente a la referencia, pero la precisión sigue siendo baja. En los ejemplos se ven predicciones sobre estructuras alargadas más extensas que los tramos anotados, fragmentación y omisiones; la línea base también responde a bordes del campo de visión y artefactos. No se ajustaron modelos ni umbrales después de ver prueba. Los resultados deben interpretarse junto con las limitaciones de etiquetas y procedencia descritas más abajo.

Se verificó además la inferencia de U-Net en las primeras **32 imágenes de `subset_10GB` por orden de nombre**: 4.16 segundos en total, aproximadamente 0.130 s/imagen, incluyendo lectura, normalización, predicción, restauración de tamaño y guardado. Las máscaras y el registro están en `outputs/subset_smoke32/`. Es una comprobación pequeña de funcionamiento y tiempo, sin etiquetas y sin medir calidad ni representar el rendimiento de las 3,113 imágenes completas.

## Uso

Desde este directorio:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Auditoría y particiones fijas; ejecutar antes de entrenar.
.venv/bin/python myotubes.py prepare
.venv/bin/python -m unittest -v test_myotubes.py

# Ajuste sólo con validación y entrenamiento desde cero.
.venv/bin/python myotubes.py classical
.venv/bin/python myotubes.py train

# Evaluación final, con ambos modelos ya seleccionados.
.venv/bin/python myotubes.py evaluate
```

El entorno `.venv` ya está preparado. `requirements.txt` contiene las dependencias de ejecución; `requirements-lock.txt` registra el entorno completo utilizado, incluidas las herramientas de lectura del PDF que no hacen falta para entrenar. Se recomienda Python 3.13 para reproducir ese entorno.

Predicción con el checkpoint entrenado:

```bash
.venv/bin/python myotubes.py predict \
  --input subset_10GB \
  --destination outputs/subset_predictions \
  --limit 32
```

Quitar `--limit 32` procesa toda la carpeta. Para usar la línea base, añadir `--method classical`. Se aceptan también imágenes individuales. Las máscaras de salida tienen las dimensiones originales y valores **0/255**; las máscaras suministradas para entrenamiento usan 0/1. La inferencia registra tiempo y fracción predicha de primer plano en `inference.json`. Sin etiquetas en `subset_10GB`, esos números no representan exactitud. Se rechaza sobrescribir máscaras existentes.

Las opciones globales van antes del subcomando, por ejemplo:

```bash
.venv/bin/python myotubes.py --data /ruta/vt --out /ruta/experimento train --device cpu
```

Usar otra carpeta `--out` para un experimento nuevo conserva los resultados anteriores. `train` inicia desde cero; no reanuda entrenamientos. `--device auto` selecciona CUDA, Metal/MPS o CPU, en ese orden.

## Decisiones del primer modelo

- Entrada gris, normalizada por sus propios percentiles 1 y 99, recortada a [0,1]. No se calculan estadísticas a partir de validación o prueba. Imágenes constantes se convierten en cero.
- Reducción a **512×384** conservando la relación 4:3 de los datos etiquetados. Interpolación de área para imagen y vecino más cercano para máscara. La entrada normalizada se almacena con precisión de 8 bits en caché para reducir memoria y lectura repetida.
- La línea base detecta regiones oscuras respecto a su entorno: suavizado 3×3, diferencia respecto a un fondo gaussiano con sigma 9, umbral, cierre 3×3 y filtrado de componentes por área y elongación. Se comparan 12 configuraciones exclusivamente en validación; `classical.json` guarda la seleccionada. Las escalas y áreas se refieren a la resolución reducida.
- U-Net con tres reducciones, canales 12/24/48/96, dos convoluciones por bloque, GroupNorm y ReLU. Decodificación bilineal con conexiones de salto y un logit por píxel. GroupNorm permite lotes pequeños sin depender de estadísticas del lote.
- Pérdida: 0.5 BCE ponderada + 0.5 Dice suave por imagen. El peso positivo es la raíz de la razón fondo/primer plano de **entrenamiento**, limitado a 10. El suavizado Dice de 1 también contempla máscaras vacías.
- Adam, tasa inicial 0.001, lote 4, máximo 40 épocas y parada tras 10 épocas sin mejora. Reducción de tasa a la mitad tras 4 épocas sin mejora. Aumentos de entrenamiento: volteos horizontales/verticales conjuntos y variaciones leves de ganancia y brillo sólo en imagen.
- Se seleccionan checkpoint y umbral por **Dice macro en las imágenes de validación con máscara no vacía**, usando umbrales de 0.1 a 0.9. Cada imagen positiva tiene el mismo peso; así las tres máscaras de área excepcionalmente grande no dominan la selección. Esta elección evita favorecer la predicción de puro fondo por acertar imágenes vacías. Los negativos se conservan en entrenamiento y evaluación, y sus falsos positivos se reportan aparte. Prueba se consulta al terminar y no decide hiperparámetros.
- Predicción en resolución de trabajo; la máscara binaria se devuelve a resolución original mediante vecino más cercano. No separa instancias ni estima longitudes biológicas.

## Datos y particiones

Hay 306 pares imagen–máscara, todos de 2592×1944. Se conservan los negativos: 34 máscaras vacías. El primer plano ocupa en promedio 1.76% de la imagen, por lo que la exactitud global sería poco informativa.

Se usa semilla 42 y se guardan 214 imágenes para entrenamiento, 46 para validación y 46 para prueba. Se agrupa por UUID de origen presente después del prefijo de exportación. Todas las imágenes `cellN` se agrupan juntas porque no conocemos su procedencia biológica; también se unen grupos que comparten imágenes exactamente iguales después de decodificarlas (12 duplicados). La asignación busca aproximar 70/15/15 usando sólo tamaños de grupos, sin mirar etiquetas ni resultados. Los grupos y hashes quedan en `outputs/manifest.csv`.

**Límite de esta partición:** no hay metadatos de placa, día ni muestra que permitan demostrar independencia biológica. Agrupar por nombre y duplicados reduce la fuga de datos detectable, pero no la descarta por completo.

### Calidad de las anotaciones suministradas

No se corrigieron ni excluyeron imágenes. La auditoría deja constancia de:

- Los polígonos originales tienen todos cuatro vértices. En la muestra visual varios son rectángulos rotados sobre tramos del miotubo, sin seguir todo su contorno. El objetivo aprendido es una aproximación a esas etiquetas.
- Tres máscaras abarcan más del 20% de la imagen; algunas son grandes rectángulos que incluyen fondo. Sus nombres están en `outputs/audit.json`.
- Hay 12 archivos de etiquetas con sufijo `(1)` que no tienen imagen homónima. No pueden constituir pares de entrenamiento y se registran como huérfanos; no se borran ni se asignan a otra imagen.
- Once grupos de imágenes exactamente iguales tienen máscaras distintas entre sus copias. Se conservan tal como se entregaron y permanecen en una misma partición; esto evita que una copia se vea en entrenamiento y otra en prueba, aunque no elimina la inconsistencia de supervisión.
- En 28 imágenes el número de instancias COCO difiere del de las etiquetas originales. Reconstruir todos los polígonos originales difiere de la máscara suministrada hasta en 2.96% de los píxeles de una imagen. Hay coordenadas negativas muy pequeñas cerca del borde en varias etiquetas afectadas, aunque no se dispone del convertidor para confirmar su causa.
- Las máscaras suministradas y los polígonos **COCO** sí concuerdan salvo diferencias pequeñas de rasterización (máximo 0.033% de los píxeles). Se usa `masks_png` como fuente de verdad, tal como recomienda el proyecto; no se regeneran máscaras silenciosamente.

Estas limitaciones afectan tanto a la línea base como a U-Net. Las métricas miden concordancia con las máscaras disponibles; no certifican una segmentación biológica completa.

## Archivos producidos

| Archivo en `outputs/` | Contenido |
|---|---|
| `unet.pt` | Pesos, configuración, época y umbral del mejor modelo |
| `classical.json` | Parámetros y validación de la línea base |
| `train.txt`, `val.txt`, `test.txt` | Particiones fijas por nombre |
| `manifest.csv`, `audit.json` | Procedencia inferida, hashes y auditoría |
| `training_config.json`, `history.json` | Configuración y evolución del entrenamiento |
| `test_metrics.json`, `test_per_image.csv` | Métricas finales agregadas y por imagen |
| `examples.jpg` | Primeros seis ejemplos de prueba por orden de nombre |

`evaluate` informa IoU, Dice, precisión y sensibilidad, tanto agregadas por píxel (**micro**) como promediadas por imagen (**macro**), en resolución de trabajo y original. También incluye **macro_nonempty**, que promedia únicamente imágenes con primer plano real, y cuenta los negativos con falsos positivos. Una predicción vacía sobre una máscara vacía tiene Dice/IoU 1; sobre una máscara no vacía tiene 0. La precisión se define como 0 cuando no hay predicciones positivas pero sí primer plano real, y como 1 si ambos están vacíos. La sensibilidad de una máscara real vacía se define como 1. Se documentan estas convenciones para interpretar promedios macro.

Los tiempos de `evaluate` excluyen lectura, preprocesamiento y guardado; los tiempos del comando `predict` incluyen esas operaciones. Las mediciones son locales y pueden variar con carga del equipo. Los originales y overlays de `verification/` nunca se usan como entrada de entrenamiento.

Las seis pruebas automatizadas comprueban aislamiento de grupos/duplicados, correspondencia de máscaras con COCO, métricas con casos vacíos, selección de umbral sin favorecer predicciones vacías, dimensiones de salida de la red y normalización de imágenes constantes. La semilla permite repetir la partición y controlar aleatoriedad; las operaciones MPS pueden introducir pequeñas diferencias numéricas entre entrenamientos.
