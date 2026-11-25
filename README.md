Next-Word Autocomplete
Proyecto Nano-AI del Track 6 — Next-Word Autocomplete. Es un modelo de lenguaje pequeño que recibe texto en inglés y propone las tres palabras que podrían continuar la secuencia.
> **Alcance:** el sistema autocompleta texto. No es un chatbot, no responde preguntas ni pretende comprender o razonar sobre el contenido.
Qué incluye
Modelo MLP entrenado desde pesos iniciales aleatorios.
Contexto de las últimas 3 palabras.
Vocabulario de 8,000 tokens: las 7,999 palabras más frecuentes de entrenamiento y `<unk>` (índice 0).
Tres sugerencias ordenadas por puntuación.
Inferencia con NumPy, sin PyTorch en la Raspberry Pi.
Interfaz web local y endpoint HTTP `POST /api/predict`.
Modelo final
Identificador: `c3_d30_more`.
Componente	Configuración
Contexto	3 palabras
Embedding	8,000 × 64
Entrada concatenada	192 valores
Capa oculta	2,048 unidades + ReLU
Salida	8,000 puntuaciones
Parámetros	17,299,264
Dropout de entrenamiento	0.3
El entrenamiento final se realizó con PyTorch y AdamW durante 10 épocas (learning rate 0.001, weight decay 0.01, batch de 1,024 y semilla 42). Los pesos exportados se guardan en `models/mlp_weights_c3_d30_more.npz`; su vocabulario asociado está en `models/vocab.json`. Estos dos archivos deben mantenerse juntos.
La Raspberry Pi utiliza `src/serve/predictor.py` para cargar los pesos y realizar la inferencia exclusivamente con NumPy.
Dataset
Los textos provienen de libros catalogados por Project Gutenberg. El manifiesto es la fuente de verdad para el título, autor, ID y partición:
44 libros de entrenamiento: 5,234,106 palabras.
2 libros de validación (Oscar Wilde): 96,109 palabras.
4 libros de prueba (Mark Twain y Jack London): 289,752 palabras.
Los autores se separan entre particiones para evaluar la generalización a escritores no vistos. El vocabulario se construye solamente con entrenamiento. La prueba final contiene objetivos `<unk>`; la métrica principal los excluye y registra cuántos se excluyeron.
Consulta data/README.md para el inventario completo, enlaces a cada libro y notas sobre las condiciones de uso de Project Gutenberg. El archivo data/manifest.csv mantiene los IDs y splits.
Resultados de evaluación
La métrica Top-3 cuenta un acierto cuando la palabra correcta aparece entre las tres sugerencias. Los ejemplos cuyo objetivo es `<unk>` se excluyen del denominador y se reportan por separado.
Prueba final en la Raspberry Pi 4
Evaluación realizada el 24 de noviembre de 2026 sobre los cuatro libros de test:
Métrica	Resultado
Top-3	26.68 %
Ejemplos evaluados	266,257
Objetivos `<unk>` excluidos	23,483
Latencia por sugerencia	29.61 ms
Memoria residente máxima del proceso	355,236 KB (≈347 MiB)
Tiempo total de evaluación	2 h 49 min 48 s
Temperatura máxima observada	68.1 °C*
Resultados por libro en la Pi:
Libro	Ejemplos evaluados	Top-3
`pg215.txt` — The Call of the Wild	28,416	28.1 %
`pg74.txt` — The Adventures of Tom Sawyer	66,279	26.5 %
`pg76.txt` — Adventures of Huckleberry Finn	104,868	25.4 %
`pg910.txt` — White Fang	66,694	28.3 %
* El monitoreo térmico empezó después de que la evaluación ya había iniciado; por ello, 68.1 °C es el máximo de las muestras disponibles, no una medición garantizada de todo el proceso. Durante el monitoreo aparecieron los estados `0x50005` y `0x50000` de `vcgencmd get_throttled`, que registran eventos de bajo voltaje y limitación de rendimiento. La fuente de alimentación utilizada no pudo identificarse. La latencia de 29.61 ms debe interpretarse como el rendimiento observado bajo esas condiciones, no como una medición libre de posibles limitaciones eléctricas.
Los archivos de evidencia de esta corrida se conservaron en `logs/` en la Raspberry Pi: `final_test_pi.json`, `pi_salida.txt`, `pi_memoria.txt`, `pi_entorno.txt` y `pi_temp.txt`. Los registros de salida y memoria se reconstruyeron a partir de la salida de terminal conservada después de finalizar el proceso; el registro del entorno también es posterior a la prueba.
Referencia de laptop
En Windows, la misma evaluación registró 26.68 % Top-3 sobre 266,257 ejemplos (23,483 excluidos) y 2.42 ms por sugerencia. El baseline de conteo obtuvo 25.81 % en ese test de laptop. El baseline no se ejecutó en la Raspberry Pi.
Ejecutar la demo
Requisitos para inferencia
Python 3.
NumPy.
Los archivos `models/mlp_weights_c3_d30_more.npz` y `models/vocab.json`.
En Windows, desde la raíz del repositorio:
```powershell
python -m pip install numpy
python -m src.serve.web_app
```
En Raspberry Pi OS de 64 bits:
```bash
sudo apt update
sudo apt install -y python3-numpy
python3 -m src.serve.web_app
```
El servidor anuncia la dirección local, normalmente:
`http://127.0.0.1:8000`
La aplicación escucha por defecto en `127.0.0.1`; no está diseñada para exponerse directamente a Internet. Para mostrar la demo de la Raspberry desde una laptop, abre un túnel SSH en la laptop:
```bash
ssh -L 8000:127.0.0.1:8000 <usuario>@<raspberry-host>.local
```
Luego inicia el servidor en la Raspberry desde la raíz del repositorio y abre `http://127.0.0.1:8000` en el navegador de la laptop.
API de predicción
Solicitud:
```http
POST /api/predict
Content-Type: application/json

{"text":"i do not"}
```
Respuesta de ejemplo:
```json
{
  "suggestions": ["know", "think", "mean"],
  "context": [
    {"word": "i", "known": true},
    {"word": "do", "known": true},
    {"word": "not", "known": true}
  ],
  "latency_ms": 2.4
}
```
El valor de latencia de este ejemplo es ilustrativo; cada respuesta contiene la medición real de esa solicitud.
Reproducir el pipeline de entrenamiento y evaluación
Ejecutar los comandos desde la raíz del repositorio y con las dependencias correspondientes instaladas. PyTorch se necesita para el entrenamiento; la inferencia en la Raspberry requiere únicamente NumPy.
```bash
python scripts/prepare_data.py
python scripts/check_splits.py
python scripts/build_dataset.py 3
python scripts/baseline.py
python scripts/train.py --dropout 0.3 --wd 0.01 --epochs 10 --tag c3_d30_more
python scripts/check_export.py
python scripts/final_eval.py --baseline --nombre laptop
```
El entrenamiento debe hacerse en un equipo adecuado; no es necesario entrenar en la Raspberry Pi. Mantén fijo el modelo final y no modifiques el conjunto de test después de consultar sus resultados.
Estructura principal
```text
data/
  manifest.csv        # ID, título, autor y split
  raw/                # textos fuente
  processed/          # textos limpios y arreglos del dataset
  README.md           # inventario y procedencia de los libros
models/
  mlp_weights_c3_d30_more.npz
  vocab.json
src/
  data/               # limpieza, normalización y ventanas
  model/              # arquitectura MLP y baseline
  eval/               # métrica Top-3
  serve/              # predictor NumPy y servidor HTTP
scripts/              # preparación, entrenamiento y evaluación
web/                  # interfaz de navegador
logs/                 # resultados y registros experimentales
PROJECT_STATUS.md     # estado y cierre técnico del proyecto
```
Limitaciones conocidas
El modelo usa únicamente las últimas tres palabras.
Su vocabulario es cerrado: las palabras desconocidas se convierten en `<unk>` y nunca se sugieren.
El corpus está formado por literatura en inglés, principalmente clásica; el rendimiento puede bajar con lenguaje moderno, jerga, nombres propios y dialectos.
La métrica Top-3 reportada excluye objetivos `<unk>`; siempre debe leerse junto con el número excluido.
No responde preguntas ni sostiene conversaciones.
Reproducibilidad y uso de datos
No sustituyas los pesos, el vocabulario ni los libros de test después de la evaluación final. Registra las condiciones de hardware y cualquier advertencia de ejecución junto con cada medición. Consulta PROJECT_STATUS.md para el cierre técnico, los resultados y las limitaciones de la corrida en Raspberry Pi.
