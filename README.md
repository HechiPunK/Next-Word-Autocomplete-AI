# Next-Word-Autocomplete-AI

## Demo web local

La demo presenta sugerencias en vivo del modelo final `c3_d30_more`. El servidor
carga los pesos y el vocabulario locales, y hace la inferencia con NumPy; no usa
PyTorch ni servicios externos para predecir.

1. Instala NumPy si aún no está instalado: `python -m pip install numpy`.
2. Desde la raíz del proyecto, inicia el servidor:

   ```powershell
   python -m src.serve.web_app
   ```

3. Abre <http://127.0.0.1:8000>.

Para usar otro puerto, ejecuta `python -m src.serve.web_app --port 8080`.
Escribe texto en inglés y selecciona una de las tres sugerencias para continuar
la frase. El modelo solo considera las últimas tres palabras, tiene un vocabulario
cerrado de 8.000 palabras y fue entrenado con literatura inglesa del siglo XIX.
El porcentaje top-3 mostrado corresponde al test separado documentado en
`logs/final_test_laptop.json`; no representa una garantía para cada entrada.
