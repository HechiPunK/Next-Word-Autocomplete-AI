# Nano-AI Project — Next-Word Autocomplete: estado del proyecto

> Última actualización: 2026-11-23
> Documento de contexto para todo el equipo que colabore en este repositorio. Léelo completo antes de cambiar código, datos o modelos.

---

## 1. Resumen en 30 segundos

- **Qué es:** un modelo de lenguaje pequeño, **entrenado desde cero** (pesos aleatorios), que sugiere **las 3 palabras más probables** que siguen a un texto en **inglés**. Es un autocompletador, no un chatbot.
- **Para qué:** proyecto universitario (track 6, *Next-Word Autocomplete*) que debe correr en una **Raspberry Pi** (unos 4 GB de RAM).
- **Estado:** el **modelo está entrenado, evaluado en la laptop y congelado**. El despliegue en la Raspberry Pi y la demo web HTTP están completados. Quedan pendientes la evaluación formal en la Pi, las mediciones de latencia y memoria, los casos reales de falla, el reporte de 2 páginas y la documentación final.
- **Modelo final:** `c3_d30_more` (contexto de 3 palabras, ~17.3 M de parámetros, 10,048 neuronas).
- **Resultado en test** (libros que el modelo nunca vio): **26.7 % top-3**, contra 25.8 % de un baseline de conteo.

---

## 2. Dónde vamos (plan de 14 días)

| Día del plan | Tarea | Estado |
|---|---|---|
| 1 | Repo y estructura de carpetas | ✅ |
| 2 | Corpus de Project Gutenberg y limpieza | ✅ |
| 3 | Vocabulario y ventanas de contexto | ✅ |
| 4 | Splits sin fuga de datos y métrica top-3 | ✅ |
| 5 | Baseline de conteo (n-gramas) | ✅ |
| 6-7 | Red neuronal (código a mano en NumPy, verificado) | ✅ |
| 8 | Experimentos y ajuste con PyTorch/GPU | ✅ |
| 9 | Exportar pesos a NumPy y `predictor.py` | ✅ |
| 12 (adelantado) | Examen final en test, corrido en la laptop | ✅ |
| 10 | Montar el modelo en la Raspberry Pi | ✅ |
| 11 | Demostración web HTTP en vivo | ✅ |
| 12 | Examen formal en la Pi, medición de latencia y memoria, y tabla de casos donde falla | ⏳ |
| 13 | Reporte de 2 páginas | ⏳ |
| 14 | Pulido, README, entrega | ⏳ |

Equivale aproximadamente al **día 11 del plan**, con el examen de test ya hecho en la laptop.

---

## 3. Reglas del proyecto (rúbrica)

**Afirmaciones permitidas**

- "Nuestro modelo alcanza X % sobre N ejemplos de test que nunca vio, corriendo en una Raspberry Pi."
- "Lo entrenamos desde pesos iniciales aleatorios, con un dataset que recolectamos."
- "No maneja Y, y esto es lo que vimos cuando falló."

**Afirmaciones prohibidas**

- Que entiende, razona o responde preguntas en general. **Solo autocompleta.**
- Cualquier número medido con datos con los que el modelo entrenó.
- Cualquier salida que no se haya producido en vivo o guardado como archivo.

**Reglas de datos**

- Solo texto escrito por el equipo o de dominio público (Project Gutenberg). Sin mensajes privados ni datos personales.
- Todo lo que se reporte debe tener un archivo en `logs/` que lo respalde.

La rúbrica del profesor no exige un protocolo específico. La demostración en vivo ya funciona por HTTP.

> Pendiente confirmar con el instructor cómo cuenta las neuronas: aquí se cuentan **capa oculta + capa de salida** (2,048 + 8,000 = 10,048). Los embeddings son una tabla de búsqueda y no se cuentan.

---

## 4. Datos

- **Fuente:** libros de Project Gutenberg en inglés (texto plano UTF-8). Lista completa con título, autor, ID y split en `data/manifest.csv`. **El manifest manda**: es la única fuente de verdad de qué libro va en qué split.
- **División por autor** (ningún autor aparece en dos splits):

| Split | Libros | Autores | Palabras |
|---|---|---|---|
| train | 44 | Austen, Dickens, Conan Doyle, Wells, Stevenson, Shelley, Stoker, Melville, Kipling, Carroll, Barrie, Brontë (Emily y Charlotte), Hardy, Hawthorne, Eliot, Conrad, Grahame, Gilman, Haggard, Alcott, Ober, Wynne | 5,234,106 |
| val | 2 | Oscar Wilde | 96,109 |
| test | 4 | Mark Twain, Jack London | 289,752 |

- **Verificación de fuga** (`scripts/check_splits.py`): sin IDs repetidos, sin autores en varios splits, y solapamiento de 8-gramas con train de **0.03 % o menos** en val y test.
- **Preprocesamiento** (`src/data/clean.py`): quita la licencia de Gutenberg, encabezados de capítulo, pasa a minúsculas y deja solo letras y apóstrofes.
- **Vocabulario:** las 7,999 palabras más frecuentes **de train** + `<unk>` (índice **0**) = **8,000**. Se arma solo con train.
- **Ejemplos** (contexto 3 → palabra siguiente): train 5,233,974 · val 96,103 · test 289,740. Porcentaje de objetivos `<unk>`: 5.7 % / 6.1 % / 8.1 %.
- **Corpus de estilo literario del siglo XIX** (ver limitaciones).

---

## 5. Pipeline y scripts

Todos se corren desde la raíz del repo.

| Orden | Script | Qué hace |
|---|---|---|
| 1 | `scripts/prepare_data.py` | Lee `data/manifest.csv`, limpia los libros de `data/raw/` y los guarda en `data/processed/{train,val,test}/` |
| 2 | `scripts/check_splits.py` | Verifica que no haya fuga de datos entre splits |
| 3 | `scripts/build_dataset.py 3` | Arma vocabulario y ventanas; guarda `X_*.npy`, `y_*.npy` y `vocab.json` en `data/processed/` |
| 4 | `scripts/baseline.py` | Baseline de n-gramas; guarda `logs/baseline_val.json` |
| 5 | `scripts/train.py` | Entrena la red (PyTorch + GPU); guarda pesos y logs |
| 6 | `scripts/check_export.py` | Comprueba que el modelo exportado en NumPy reproduce el resultado de PyTorch |
| 7 | `scripts/final_eval.py` | Examen final en test (red y, opcionalmente, baseline); guarda `logs/final_test_<nombre>.json` |

**Métrica (`src/eval/metrics.py`):** *top-3 accuracy* = la palabra correcta está entre las 3 sugeridas. Los ejemplos cuyo objetivo es `<unk>` se **excluyen** (y se reporta cuántos), y el modelo **nunca sugiere `<unk>`**.

---

## 6. Modelo final

**Nombre:** `c3_d30_more`

**Arquitectura (MLP, estilo Bengio):**

```
3 palabras (IDs) ─► Embedding (8000 × 64) ─► concatenar (192)
                 ─► Linear 192→2048 + ReLU (+ Dropout 0.3 solo al entrenar)
                 ─► Linear 2048→8000 ─► softmax ─► top-3
```

- **Parámetros:** 17,299,264 · **Neuronas:** 2,048 (oculta) + 8,000 (salida) = **10,048**.
- **Entrenamiento:** PyTorch en GPU (RTX 5050), AdamW, lr 1e-3, weight decay 0.01, batch 1024, dropout 0.3, 10 épocas, semilla 42. El mejor top-3 en val fue 29.8 % (época 10).
- **Comando:** `python scripts/train.py --dropout 0.3 --wd 0.01 --epochs 10 --tag c3_d30_more` (con el dataset en **contexto 3**). Reproducir puede dar variaciones pequeñas por la GPU.

**Archivos del modelo (carpeta `models/`):**

| Archivo | Descripción |
|---|---|
| `mlp_weights_c3_d30_more.npz` | Pesos finales: `E` (8000×64), `W1` (192×2048), `b1`, `W2` (2048×8000), `b2` |
| `vocab.json` | Lista de 8,000 palabras; la posición es el ID. **Siempre va en pareja con los pesos** |

El checkpoint original de PyTorch (`best_c3_d30_more.pt`) se conserva **solo en la laptop de desarrollo**, ignorado por Git. Los modelos de los demás experimentos fueron eliminados a propósito.

**Cómo se usa (solo NumPy, es lo que corre en la Pi):**

```python
from src.serve.predictor import Predictor

p = Predictor("models/mlp_weights_c3_d30_more.npz", "models/vocab.json")
p.predict_top3("i do not")   # -> ['know', 'think', 'mean']
```

`Predictor.predict_top3(texto)` hace: normaliza el texto con `src.data.clean.normalize` → toma las **últimas 3 palabras** (si hay menos, rellena con `<unk>`) → red → devuelve las 3 con mayor puntaje, nunca `<unk>`. **Esta es la interfaz** que usa el servidor HTTP.

---

## 7. Resultados

### Examen final en test (modelo congelado, `logs/final_test_laptop.json`)

| Modelo | top-3 val | top-3 test |
|---|---|---|
| Baseline de conteo (orden 3, `min_count` 10) | 28.9 % | 25.8 % |
| **Red neuronal (modelo final)** | **29.8 %** | **26.7 %** (± 0.2 %*) |

\* Margen de 95 % aproximado que supone ejemplos independientes; la incertidumbre real es algo mayor.

- **N evaluados en test:** 266,257. **Excluidos por `<unk>`:** 23,483 (8.1 %).
- **Medida más estricta** (contando los `<unk>` como fallos): ≈ 24.5 % la red y ≈ 23.7 % el baseline.
- **Por libro (red / baseline):**

| Libro | Red | Baseline |
|---|---|---|
| *The Call of the Wild* (London) | 28.1 % | 27.3 % |
| *White Fang* (London) | 28.3 % | 27.4 % |
| *The Adventures of Tom Sawyer* (Twain) | 26.5 % | 25.7 % |
| *Adventures of Huckleberry Finn* (Twain) | 25.4 % | 24.5 % |

- **Velocidad en la laptop (Windows):** 2.4 ms por sugerencia. **En la Raspberry:** siguen pendientes la evaluación formal (`logs/final_test_pi.json`) y las mediciones de latencia y memoria.

### Experimentos (top-3 en val; respaldados por `logs/experiments.csv` y `logs/train_log_*.csv`)

| Modelo | Datos de train | Contexto | Dropout | top-3 val |
|---|---|---|---|---|
| Baseline de conteo | 2.3 M palabras | 3 | – | 27.6 % |
| Red | 2.3 M | 3 | 0.2 | 27.6 % |
| Red | 2.3 M | 3 | 0.5 | 28.1 % |
| Red | 2.3 M | 5 | 0.5 | 27.8 % |
| Baseline de conteo | 5.2 M | 3 | – | 28.9 % |
| Red | 5.2 M | 3 | 0.5 | 28.7 % |
| **Red final** | **5.2 M** | **3** | **0.3** | **29.8 %** |
| Red | 5.2 M | 5 | 0.3 | 29.8 % |
| Red (embedding 40) | 5.2 M | 5 | 0.4 | 29.6 % |

**Conclusiones:** (1) más datos ayudó a todos los modelos; (2) con 2.3 M de palabras había que frenar mucho la red (dropout 0.5) y con 5.2 M ya no; (3) un contexto de 5 palabras no mejoró, y el modelo parece limitado por la cantidad de datos más que por el diseño; (4) la red supera al baseline por ~0.9 puntos de forma consistente en val, en test y en los 4 libros de test.

---

## 8. Limitaciones conocidas (para el reporte)

- **Solo mira las últimas 3 palabras.** No usa contexto más largo (ej. `it was a dark and` se reduce a `a dark and`).
- **Entrenado con literatura inglesa del siglo XIX**: funciona peor con texto moderno, jerga o escritura informal.
- **Vocabulario cerrado de 8,000 palabras:** nombres propios y palabras raras caen en `<unk>` y nunca se sugieren.
- **Dialecto:** el peor libro de test es *Huckleberry Finn* (25.4 %), escrito en dialecto.
- **No entiende ni razona:** solo predice qué palabras suelen seguir a otras tres.
- **Casos de falla con ejemplos reales:** pendiente (deben salir de una corrida en vivo, guardada en `logs/`).

---

## 9. Cómo reproducir todo

```powershell
python scripts/prepare_data.py
python scripts/check_splits.py
python scripts/build_dataset.py 3
python scripts/baseline.py
python scripts/train.py --dropout 0.3 --wd 0.01 --epochs 10 --tag c3_d30_more
python scripts/check_export.py
python scripts/final_eval.py --baseline --nombre laptop
```

Requisitos: Python 3, NumPy; **PyTorch con CUDA solo para entrenar** (para una RTX serie 50 se necesita una versión de PyTorch compilada para CUDA 12.8 o superior). Entrenar y evaluar **no hace falta repetirlo** para montar el proyecto.

---

## 10. Cómo montarlo en la Raspberry Pi

**Equipo:** Raspberry Pi 4 Model B con 4 GB de RAM. **Sistema:** Raspberry Pi OS de 64 bits (Debian 13 Trixie, arquitectura `aarch64`). **Versiones:** Python 3.13.5, NumPy 2.2.4 y Git 2.47.3.

1. Se instalaron Git y NumPy mediante `apt` (la Pi solo necesita NumPy, **no PyTorch**):
   ```bash
   sudo apt update && sudo apt install -y python3-numpy git
   ```
2. Se clonó el repositorio con *sparse checkout*, incluyendo `/src/**`, `/web/**`, `models/mlp_weights_c3_d30_more.npz` y `models/vocab.json`. El historial arrastra modelos viejos pesados. Se actualizó la copia de la Pi con `git pull --ff-only origin main` y quedó en el commit `ea2910b` (`interfaz`).
3. Se verificó la presencia de los pesos `models/mlp_weights_c3_d30_more.npz` (aprox. 66 MB) y del vocabulario `models/vocab.json` (aprox. 84 KB).
4. Se ejecutó correctamente `Predictor` en la Pi y se obtuvieron estas sugerencias:

   | Texto | Sugerencias |
   |---|---|
   | `i do not` | `know`, `think`, `mean` |
   | `it was a dark and` | `a`, `the`, `sinister` |
   | `she looked at him with` | `a`, `the`, `an` |

5. Se abrió y probó la **demostración web HTTP** desde Windows. `src/serve/web_app.py` implementa el servidor con `ThreadingHTTPServer` y escucha en `127.0.0.1:8000`. `web/app.js` usa `fetch()` para enviar JSON a `POST /api/predict`, que devuelve sugerencias, contexto y latencia. Para reproducir la demo:

   - En Windows, abrir el túnel SSH:
     ```bash
     ssh -L 8000:127.0.0.1:8000 sebas@sebas-raspberry.local
     ```
   - En la sesión SSH, desde la raíz del repo en la Pi, ejecutar `python3 -m src.serve.web_app`.
   - En el navegador de Windows, abrir `http://127.0.0.1:8000`.

6. **Evaluación formal pendiente.** Falta copiar a la Pi `data/processed/test/` (menos de 2 MB, ignorada por Git) y los scripts de evaluación necesarios, incluido `scripts/final_eval.py`, porque el *sparse checkout* actual no los incluye. Después, ejecutar `python3 scripts/final_eval.py --nombre pi` para generar `logs/final_test_pi.json`, comparar el top-3 con el de la laptop y registrar la latencia. Queda pendiente medir también el uso de memoria.

---

## 11. Pendientes y roles sugeridos

| Rol | Responsable | Pendiente |
|---|---|---|
| Datos | _por asignar_ | `data/README.md` con título, autor, ID y licencia (dominio público) de cada libro |
| Modelo | _por asignar_ | Nada: modelo congelado |
| Pi / Servidor | _por asignar_ | Copiar el conjunto de test y los scripts de evaluación a la Pi, ejecutar `scripts/final_eval.py` y medir latencia y memoria |
| Evaluación / Reporte | _por asignar_ | Tabla de casos donde falla (en vivo, guardada), reporte de 2 páginas, README final |

---

## 12. Reglas para quien colabore (personas e IAs)

1. **El modelo está congelado.** No reentrenar ni reemplazar `mlp_weights_c3_d30_more.npz` sin avisar al equipo.
2. **El test no se toca.** No cambiar libros del test ni del manifest, y no tomar decisiones de diseño mirando resultados de test. Cualquier cambio posterior invalida el resultado final.
3. **Pesos y vocabulario van en pareja.** Si cambia `data/manifest.csv` (libros de train), cambia el vocabulario y los pesos actuales dejan de servir.
4. **Contexto = 3.** `build_dataset.py N` **sobrescribe** los `.npy`. Antes de evaluar, el dataset debe estar en contexto 3.
5. **Convenciones fijas:** `<unk>` es el ID 0; nunca se sugiere `<unk>`; el texto se normaliza siempre con `src/data/clean.normalize`; la métrica es top-3 sin contar objetivos `<unk>`.
6. **La Raspberry solo usa NumPy.** Nada de PyTorch en `src/serve/`.
7. **Cada número del reporte necesita un archivo en `logs/`.** No inventar ni redondear a favor.
8. **No subir al repo** pesos de experimentos, checkpoints `.pt` ni `data/processed/` (ya están en `.gitignore`).
9. **Mensajes de commit** con prefijos `feat:`, `fix:`, `docs:`, `chore:`; ramas por tarea (`feature/...`) y Pull Request a `main`.

---

## 13. Estructura del repositorio

```
├── README.md
├── PROJECT_STATUS.md        # este documento
├── requirements.txt         # numpy (lo único que necesita la Pi)
├── data/
│   ├── manifest.csv         # libros, autores y split (fuente de verdad)
│   ├── raw/                 # libros originales de Gutenberg
│   └── processed/           # generado por los scripts (ignorado por Git)
├── src/
│   ├── data/                # clean.py (limpieza/normalización), dataset.py (vocabulario y ventanas)
│   ├── model/               # mlp.py (red a mano en NumPy), ngram_baseline.py
│   ├── eval/                # metrics.py (top-3)
│   └── serve/               # predictor.py (inferencia en NumPy), web_app.py (servidor HTTP)
├── web/                     # frontend de la demostración web
├── scripts/                 # prepare_data, check_splits, build_dataset, baseline, train, check_export, final_eval
├── models/                  # mlp_weights_c3_d30_more.npz + vocab.json (modelo final)
├── logs/                    # evidencia: experimentos, baseline, resultados finales
├── examples/                # neuron.py, network2cells.py, regression.py (base del curso)
└── docs/
```
