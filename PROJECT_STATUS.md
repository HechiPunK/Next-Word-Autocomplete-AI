# Next-Word Autocomplete — estado final del proyecto

> Actualizado: 24 de noviembre de 2026  
> Repositorio: `HechiPunK/Next-Word-Autocomplete-AI`  
> Rama revisada: `main` — commit de referencia `ab01a314054f0d3976c851caaa3334176387a6eb` (`ab01a31`, mensaje `Dinosaurio`).

## 1. Resumen ejecutivo

El proyecto Track 6, **Next-Word Autocomplete**, llegó al cierre de su desarrollo técnico. Se entrenó desde pesos aleatorios un modelo MLP para sugerir las tres palabras siguientes más probables en inglés, se exportó a pesos NumPy, se desplegó en una Raspberry Pi 4 Model B y se verificó una demostración web local mediante HTTP. La evaluación final en la Pi terminó correctamente y produjo una precisión Top-3 de **26.68 % en 266,257 ejemplos evaluados**.

El producto es un **autocompletador**, no un chatbot: no responde preguntas, no sostiene conversaciones ni afirma comprender el texto.

### Estado de los entregables de la rúbrica

| Entregable solicitado | Estado técnico | Evidencia / actividad de cierre |
|---|---|---|
| Demo en vivo | Implementada y probada | Interfaz web local servida desde la Raspberry Pi; ver sección 6. Debe abrirse nuevamente durante la presentación. |
| Dataset | Integrado y documentado | `data/manifest.csv` y `data/README.md`; 50 libros con título, autor, ID y enlace de origen. |
| Código | Implementado | Scripts, `src/`, `web/` y archivos del modelo en el repositorio. Se restaura este README para documentar instalación y ejecución. |
| Reporte técnico de dos páginas | Redactado | Informe técnico preparado en Word; falta agregar portada/datos de equipo antes de entregar. |

La rúbrica pide una cifra medida en datos que el modelo nunca vio y solicita entregar demo, dataset, código y un reporte técnico de dos páginas.

## 2. Dataset y particiones

**Fuente:** libros de Project Gutenberg en inglés. El inventario completo y los enlaces por eBook están en [`data/README.md`](data/README.md); el manifiesto [`data/manifest.csv`](data/manifest.csv) es la fuente de verdad para ID, título, autor y split.

| Split | Libros | Autores | Palabras | Ejemplos de contexto 3 → siguiente |
|---|---:|---|---:|---:|
| train | 44 | 23 autores | 5,234,106 | 5,233,974 |
| val | 2 | Oscar Wilde | 96,109 | 96,103 |
| test | 4 | Mark Twain y Jack London | 289,752 | 289,740 |
| **Total** | **50** | **26 autores** | **5,619,967** | **5,619,817** |

- Ningún autor aparece en más de una partición según el manifiesto.
- `scripts/check_splits.py` comprueba IDs repetidos, autores asignados a varios splits y solapamiento de 8-gramas entre train y val/test.
- El vocabulario contiene 8,000 entradas: las 7,999 palabras más frecuentes de train más `<unk>` (ID 0). No se construye con validación ni test.
- La métrica Top-3 excluye objetivos `<unk>` y reporta el número excluido.
- El preprocesamiento está implementado en `src/data/clean.py`: retira marcadores de licencia de Gutenberg cuando los encuentra, elimina encabezados reconocidos, convierte a minúsculas y conserva letras inglesas, apóstrofes y espacios.

**Nota de procedencia/derechos:** Project Gutenberg indica que su evaluación de copyright se basa en la legislación estadounidense y aconseja a usuarios de otros países comprobar la legislación local. El inventario enlaza las fichas de origen, pero no sustituye una revisión jurídica individual de cada eBook para su redistribución en México u otra jurisdicción. Ver los enlaces de términos y permisos en `data/README.md`.

## 3. Modelo final y entrenamiento

**Modelo congelado:** `c3_d30_more`.

| Componente | Configuración |
|---|---:|
| Contexto | 3 palabras |
| Vocabulario | 8,000 tokens |
| Embedding | 8,000 × 64 |
| Vector de entrada concatenado | 192 |
| Capa oculta | 2,048 unidades, ReLU |
| Capa de salida | 8,000 puntuaciones |
| Parámetros | 17,299,264 |

El entrenamiento final se realizó con `scripts/train.py` usando PyTorch, AdamW, learning rate 0.001, weight decay 0.01, dropout 0.3, batch 1,024, semilla 42 y 10 épocas. El mejor Top-3 de validación fue 29.84 % (29.8 % redondeado) en la época 10. El archivo exportado para inferencia es `models/mlp_weights_c3_d30_more.npz`, emparejado con `models/vocab.json`.

La Raspberry Pi ejecuta `src/serve/predictor.py` con NumPy. PyTorch se usa para el entrenamiento en el equipo de desarrollo, no para inferencia en la Pi.

## 4. Evaluación de laptop y baseline

El archivo `logs/final_test_laptop.json` registra la evaluación en Windows del 11 de noviembre de 2026:

| Modelo | Top-3 test | Ejemplos evaluados | Excluidos `<unk>` | Latencia |
|---|---:|---:|---:|---:|
| Red neuronal final | 26.68 % | 266,257 | 23,483 | 2.42 ms/sugerencia |
| Baseline de n-gramas (orden 3, min_count 10) | 25.81 % | 266,257 | 23,483 | No es la medición de latencia de la Pi |

El baseline se evaluó en laptop; **no se ejecutó como parte de la evaluación formal en Raspberry Pi**. Por tanto, la comparación red-baseline del test corresponde al equipo de desarrollo, no a una comparación realizada en la Pi.

## 5. Evaluación formal en Raspberry Pi

**Equipo:** Raspberry Pi 4 Model B Rev 1.1, 4 GB RAM.  
**Entorno documentado:** Raspberry Pi OS 64 bits / Debian 13 Trixie, `aarch64`, Python 3.13.5, NumPy 2.2.4.

La ejecución de `scripts/final_eval.py --nombre pi` finalizó con `Exit status: 0` el 24 de noviembre de 2026. Los valores obtenidos fueron:

| Métrica | Resultado observado |
|---|---:|
| Top-3 | **0.2668 (26.68 %)** |
| Ejemplos evaluados | **266,257** |
| Excluidos por objetivo `<unk>` | **23,483** |
| Latencia media de 300 sugerencias | **29.61 ms por sugerencia** |
| Tiempo de pared | **2 h 49 min 48 s** |
| CPU utilizada por el proceso | 99 % |
| Memoria residente máxima del proceso | 355,236 KB (≈347 MiB) |
| Swaps durante el proceso | 0 |

### Resultados por libro

| Archivo / obra | Autor | Ejemplos evaluados | Top-3 |
|---|---|---:|---:|
| `pg215.txt` — The Call of the Wild | Jack London | 28,416 | 28.1 % |
| `pg74.txt` — The Adventures of Tom Sawyer | Mark Twain | 66,279 | 26.5 % |
| `pg76.txt` — Adventures of Huckleberry Finn | Mark Twain | 104,868 | 25.4 % |
| `pg910.txt` — White Fang | Jack London | 66,694 | 28.3 % |

Los conteos de ejemplos evaluados y excluidos coinciden con los valores de referencia de la evaluación de laptop. El Top-3 también coincide a cuatro decimales con el resultado de referencia.

### Condiciones y advertencias de hardware

- El registro térmico comenzó después de iniciada la evaluación; la máxima de **68.1 °C** es el máximo observado en ese intervalo parcial, no un máximo garantizado de toda la ejecución.
- Durante el monitoreo se registraron `throttled=0x50005` y `throttled=0x50000`. Estos valores incluyen eventos de bajo voltaje y limitación de rendimiento. La última lectura fue `throttled=0x50000`, que conserva los indicadores históricos y no muestra los bits de evento activo en ese instante.
- No se identificaron el fabricante ni las especificaciones de la fuente/cable de alimentación; no es posible atribuir la alerta a una pieza específica.
- En consecuencia, la latencia de 29.61 ms es el valor realmente observado en esta Pi bajo las condiciones registradas. Puede haber sido afectada por la condición de alimentación; no debe presentarse como un benchmark de hardware con alimentación ideal.
- No se observó evidencia de throttling térmico en los códigos registrados. La temperatura máxima parcial tampoco sugiere sobrecalentamiento durante las muestras tomadas.

### Trazabilidad de los registros

En la Raspberry Pi quedaron en `logs/` los siguientes archivos: `final_test_pi.json`, `pi_salida.txt`, `pi_memoria.txt`, `pi_entorno.txt` y `pi_temp.txt`.

- `final_test_pi.json` fue generado por `final_eval.py`.
- `pi_temp.txt` registra las muestras tomadas desde que se inició el bucle de monitoreo; no cubre toda la evaluación.
- `pi_salida.txt` y `pi_memoria.txt` se reconstruyeron después de la prueba a partir de la salida completa de `/usr/bin/time -v` conservada en la terminal.
- `pi_entorno.txt` es un registro complementario posterior a la evaluación; no debe interpretarse como una captura de entorno previa al examen.
- Los registros Pi permanecen en el almacenamiento local de la Raspberry Pi y no se deben considerar incorporados al repositorio hasta que se copien y versionen explícitamente.
- No se realizó una comparación SHA-256 de los pesos/vocabulario entre Pi y laptop, porque no se contó con una copia local de la carpeta del proyecto en la laptop. La coincidencia de los conteos y métrica con la referencia es consistente con el mismo test, pero no sustituye una comparación criptográfica de archivos.

## 6. Despliegue y demostración

La demostración HTTP quedó implementada y fue probada desde un navegador de la laptop mediante túnel SSH.

Flujo:

`Navegador → POST /api/predict → src/serve/web_app.py → Predictor NumPy → pesos + vocabulario`

- Interfaz: `web/index.html`, `web/styles.css`, `web/app.js`.
- Servidor: `src/serve/web_app.py`, basado en `http.server` de Python.
- Endpoint: `POST /api/predict`, recibe JSON con el campo `text` y devuelve `suggestions`, `context` y `latency_ms`.
- La Pi escucha por defecto en `127.0.0.1:8000`; no exponer directamente a Internet.

Para repetir la demo durante la presentación:

1. Desde la laptop, abrir `ssh -L 8000:127.0.0.1:8000 sebas@sebas-raspberry.local` (ajustar usuario/host si cambia).
2. En la Raspberry, desde la raíz del repositorio, ejecutar `python3 -m src.serve.web_app`.
3. En el navegador de la laptop, abrir `http://127.0.0.1:8000`.
4. Escribir una frase en inglés y mostrar que el modelo propone tres continuaciones; aclarar que solo autocompleta.

## 7. Limitaciones conocidas

- Contexto fijo de las últimas tres palabras; no considera el contexto anterior.
- Vocabulario cerrado de 8,000 entradas; palabras raras, nombres propios y palabras no vistas se representan con `<unk>` y nunca se sugieren.
- Corpus de literatura inglesa clásica; el modelo puede generalizar peor a lenguaje moderno, jerga, redes sociales o dialectos.
- En la prueba, `Adventures of Huckleberry Finn` obtuvo el menor Top-3 (25.4 %), dato compatible con la dificultad del dialecto narrativo, aunque no demuestra por sí solo la causa.
- El modelo completa secuencias de texto; no entiende preguntas ni genera respuestas conversacionales.
- La latencia observada en la Pi debe contextualizarse por las alertas de bajo voltaje.

## 8. Cierre y pendientes de entrega

### Desarrollo técnico completado

- [x] Dataset de 50 libros, inventariado y separado por autor.
- [x] Limpieza, construcción de vocabulario y ventanas de contexto 3.
- [x] Comprobaciones de particiones y baseline.
- [x] Entrenamiento del modelo final y congelamiento de pesos.
- [x] Exportación y verificación de inferencia NumPy.
- [x] Despliegue en Raspberry Pi 4.
- [x] Demostración web HTTP verificada.
- [x] Evaluación final del modelo en laptop.
- [x] Evaluación formal del modelo en Raspberry Pi, con advertencias de alimentación documentadas.
- [x] Informe técnico de dos páginas redactado.
- [x] README principal y documentación del inventario del dataset preparados.

### Antes de presentar/entregar

- [ ] Agregar portada, nombres del equipo y datos de la materia al informe Word.
- [ ] Ejecutar la demostración en vivo el día de la exposición y comprobar la alimentación de la Pi.
- [ ] Copiar los cinco registros de `logs/` de la Pi al paquete final de entrega, si el equipo desea adjuntarlos como evidencia reproducible.
- [ ] Si se dispone del proyecto en la laptop, completar la comparación SHA-256 de `models/mlp_weights_c3_d30_more.npz` y `models/vocab.json`; no es necesario repetir la evaluación.
- [ ] Revisar el estatus de derechos de cada eBook conforme a la jurisdicción donde se redistribuya el dataset.

### Mejoras opcionales, no bloqueantes para la rúbrica

- Registrar de forma sistemática ejemplos cualitativos de errores de autocompletado con sus entradas y salidas observadas.
- Repetir una medición de latencia solo si se consigue una fuente de alimentación conocida y adecuada, y únicamente si el equipo decide añadir una medición complementaria claramente identificada; conservar el resultado actual sin reemplazarlo.

**Conclusión de estado:** el desarrollo técnico principal llegó al final. El proyecto tiene modelo, dataset documentado, código, demo funcional y evaluación final en Raspberry Pi. Lo que queda es preparar la presentación y reunir el paquete final de entrega, no reentrenar ni modificar el modelo.
