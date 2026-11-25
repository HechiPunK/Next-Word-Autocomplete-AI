# Inventario y procedencia del dataset

Este directorio contiene el manifiesto del corpus utilizado por el proyecto **Next-Word Autocomplete**. La lista de referencia es [`manifest.csv`](manifest.csv), con una fila por libro y los campos `id`, `title`, `author` y `split`.

## Procedencia

Los 50 títulos registrados en el manifiesto están vinculados a sus fichas de Project Gutenberg mediante el identificador de eBook. Los enlaces de esta lista permiten consultar la ficha y las descargas disponibles para cada obra.

**Nota importante sobre derechos:** Project Gutenberg indica que sus determinaciones de dominio público se realizan conforme a la legislación de Estados Unidos y recomienda a las personas fuera de ese país comprobar la legislación local. Por tanto, este inventario identifica la fuente de consulta y no constituye una certificación jurídica individual de dominio público en todos los países. Antes de redistribuir los textos o publicar copias, revisa la ficha, el encabezado/licencia del eBook y las normas aplicables en tu jurisdicción. Consulta [Terms of Use](https://www.gutenberg.org/policy/terms_of_use) y [Permissions, Licensing and other Common Requests](https://www.gutenberg.org/policy/permission.html).

## Resumen de particiones

| Partición | Libros | Autores | Uso |
|---|---:|---|---|
| `train` | 44 | 23 autores | Entrenamiento del vocabulario y del modelo |
| `val` | 2 | Oscar Wilde | Selección de configuración y seguimiento durante entrenamiento |
| `test` | 4 | Mark Twain y Jack London | Evaluación final con autores no vistos |
| **Total** | **50** | **26 autores** | |

Los autores están separados entre particiones según `manifest.csv`: no hay un mismo autor asignado a más de un split. El proyecto documenta una comprobación adicional de IDs duplicados y solapamiento de 8-gramas mediante `scripts/check_splits.py`.

## Inventario completo

### Entrenamiento (`train`) — 44 libros

| ID | Obra | Autor | Fuente |
|---:|---|---|---|
| 105 | Persuasion | Jane Austen | [Project Gutenberg eBook 105](https://www.gutenberg.org/ebooks/105) |
| 158 | Emma | Jane Austen | [Project Gutenberg eBook 158](https://www.gutenberg.org/ebooks/158) |
| 1342 | Pride and Prejudice | Jane Austen | [Project Gutenberg eBook 1342](https://www.gutenberg.org/ebooks/1342) |
| 98 | A Tale of Two Cities | Charles Dickens | [Project Gutenberg eBook 98](https://www.gutenberg.org/ebooks/98) |
| 730 | Oliver Twist | Charles Dickens | [Project Gutenberg eBook 730](https://www.gutenberg.org/ebooks/730) |
| 1400 | Great Expectations | Charles Dickens | [Project Gutenberg eBook 1400](https://www.gutenberg.org/ebooks/1400) |
| 1661 | The Adventures of Sherlock Holmes | Arthur Conan Doyle | [Project Gutenberg eBook 1661](https://www.gutenberg.org/ebooks/1661) |
| 2852 | The Hound of the Baskervilles | Arthur Conan Doyle | [Project Gutenberg eBook 2852](https://www.gutenberg.org/ebooks/2852) |
| 35 | The Time Machine | H. G. Wells | [Project Gutenberg eBook 35](https://www.gutenberg.org/ebooks/35) |
| 36 | The War of the Worlds | H. G. Wells | [Project Gutenberg eBook 36](https://www.gutenberg.org/ebooks/36) |
| 5230 | The Invisible Man | H. G. Wells | [Project Gutenberg eBook 5230](https://www.gutenberg.org/ebooks/5230) |
| 43 | Strange Case of Dr Jekyll and Mr Hyde | Robert Louis Stevenson | [Project Gutenberg eBook 43](https://www.gutenberg.org/ebooks/43) |
| 120 | Treasure Island | Robert Louis Stevenson | [Project Gutenberg eBook 120](https://www.gutenberg.org/ebooks/120) |
| 84 | Frankenstein | Mary Shelley | [Project Gutenberg eBook 84](https://www.gutenberg.org/ebooks/84) |
| 345 | Dracula | Bram Stoker | [Project Gutenberg eBook 345](https://www.gutenberg.org/ebooks/345) |
| 2701 | Moby Dick | Herman Melville | [Project Gutenberg eBook 2701](https://www.gutenberg.org/ebooks/2701) |
| 236 | The Jungle Book | Rudyard Kipling | [Project Gutenberg eBook 236](https://www.gutenberg.org/ebooks/236) |
| 11 | Alice's Adventures in Wonderland | Lewis Carroll | [Project Gutenberg eBook 11](https://www.gutenberg.org/ebooks/11) |
| 16 | Peter Pan | J. M. Barrie | [Project Gutenberg eBook 16](https://www.gutenberg.org/ebooks/16) |
| 768 | Wuthering Heights | Emily Brontë | [Project Gutenberg eBook 768](https://www.gutenberg.org/ebooks/768) |
| 27 | Far from the Madding Crowd | Thomas Hardy | [Project Gutenberg eBook 27](https://www.gutenberg.org/ebooks/27) |
| 77 | The House of the Seven Gables | Nathaniel Hawthorne | [Project Gutenberg eBook 77](https://www.gutenberg.org/ebooks/77) |
| 110 | Tess of the d'Urbervilles: A Pure Woman | Thomas Hardy | [Project Gutenberg eBook 110](https://www.gutenberg.org/ebooks/110) |
| 122 | The Return of the Native | Thomas Hardy | [Project Gutenberg eBook 122](https://www.gutenberg.org/ebooks/122) |
| 143 | The Mayor of Casterbridge | Thomas Hardy | [Project Gutenberg eBook 143](https://www.gutenberg.org/ebooks/143) |
| 145 | Middlemarch | George Eliot | [Project Gutenberg eBook 145](https://www.gutenberg.org/ebooks/145) |
| 153 | Jude the Obscure | Thomas Hardy | [Project Gutenberg eBook 153](https://www.gutenberg.org/ebooks/153) |
| 219 | Heart of Darkness | Joseph Conrad | [Project Gutenberg eBook 219](https://www.gutenberg.org/ebooks/219) |
| 224 | A Pair of Blue Eyes | Thomas Hardy | [Project Gutenberg eBook 224](https://www.gutenberg.org/ebooks/224) |
| 289 | The Wind in the Willows | Kenneth Grahame | [Project Gutenberg eBook 289](https://www.gutenberg.org/ebooks/289) |
| 507 | Adam Bede | George Eliot | [Project Gutenberg eBook 507](https://www.gutenberg.org/ebooks/507) |
| 512 | Mosses from an Old Manse | Nathaniel Hawthorne | [Project Gutenberg eBook 512](https://www.gutenberg.org/ebooks/512) |
| 550 | Silas Marner | George Eliot | [Project Gutenberg eBook 550](https://www.gutenberg.org/ebooks/550) |
| 1260 | Jane Eyre: An Autobiography | Charlotte Brontë | [Project Gutenberg eBook 1260](https://www.gutenberg.org/ebooks/1260) |
| 1952 | The Yellow Wallpaper | Charlotte Perkins Gilman | [Project Gutenberg eBook 1952](https://www.gutenberg.org/ebooks/1952) |
| 2166 | King Solomon's Mines | H. Rider Haggard | [Project Gutenberg eBook 2166](https://www.gutenberg.org/ebooks/2166) |
| 5658 | Lord Jim | Joseph Conrad | [Project Gutenberg eBook 5658](https://www.gutenberg.org/ebooks/5658) |
| 6688 | The Mill on the Floss | George Eliot | [Project Gutenberg eBook 6688](https://www.gutenberg.org/ebooks/6688) |
| 9182 | Villette | Charlotte Brontë | [Project Gutenberg eBook 9182](https://www.gutenberg.org/ebooks/9182) |
| 25344 | The Scarlet Letter | Nathaniel Hawthorne | [Project Gutenberg eBook 25344](https://www.gutenberg.org/ebooks/25344) |
| 37106 | Little Women | Louisa May Alcott | [Project Gutenberg eBook 37106](https://www.gutenberg.org/ebooks/37106) |
| 56665 | Tales and Stories | Mary Wollstonecraft Shelley | [Project Gutenberg eBook 56665](https://www.gutenberg.org/ebooks/56665) |
| 68779 | Personal reminiscences of Henry Irving | Bram Stoker | [Project Gutenberg eBook 68779](https://www.gutenberg.org/ebooks/68779) |
| 79867 | The double thirteen | Anthony Wynne | [Project Gutenberg eBook 79867](https://www.gutenberg.org/ebooks/79867) |

### Validación (`val`) — 2 libros

| ID | Obra | Autor | Fuente |
|---:|---|---|---|
| 174 | The Picture of Dorian Gray | Oscar Wilde | [Project Gutenberg eBook 174](https://www.gutenberg.org/ebooks/174) |
| 902 | The Happy Prince and Other Tales | Oscar Wilde | [Project Gutenberg eBook 902](https://www.gutenberg.org/ebooks/902) |

### Prueba (`test`) — 4 libros

| ID | Obra | Autor | Fuente |
|---:|---|---|---|
| 74 | The Adventures of Tom Sawyer | Mark Twain | [Project Gutenberg eBook 74](https://www.gutenberg.org/ebooks/74) |
| 76 | Adventures of Huckleberry Finn | Mark Twain | [Project Gutenberg eBook 76](https://www.gutenberg.org/ebooks/76) |
| 215 | The Call of the Wild | Jack London | [Project Gutenberg eBook 215](https://www.gutenberg.org/ebooks/215) |
| 910 | White Fang | Jack London | [Project Gutenberg eBook 910](https://www.gutenberg.org/ebooks/910) |

## Preprocesamiento utilizado

El código de `src/data/clean.py` realiza las siguientes transformaciones antes de formar las ventanas de entrenamiento:

1. Retira el bloque de inicio y final de Project Gutenberg cuando coincide con los marcadores esperados.
2. Elimina encabezados que comienzan con `Chapter`, `Book`, `Part`, `Volume` o `Contents`.
3. Convierte el texto a minúsculas.
4. Conserva letras inglesas (`a-z`), apóstrofes y espacios; normaliza espacios repetidos.
5. Construye el vocabulario usando únicamente los textos de `train`: 7,999 palabras frecuentes más `<unk>`.
6. Forma ejemplos con tres palabras de contexto y la palabra siguiente como objetivo.

Los textos limpios y los arreglos derivados se guardan en `data/processed/` y se generan mediante los scripts del repositorio; no deben editarse manualmente después de congelar la evaluación final.

## Conteos del corpus documentados por el proyecto

| Split | Palabras del texto limpio | Ejemplos contexto→siguiente |
|---|---:|---:|
| train | 5,234,106 | 5,233,974 |
| val | 96,109 | 96,103 |
| test | 289,752 | 289,740 |

Los objetivos `<unk>` reportados para train/val/test son aproximadamente 5.7 %, 6.1 % y 8.1 %, respectivamente. La métrica Top-3 final excluye objetivos `<unk>` y debe acompañarse siempre del número excluido.

## Fuentes y referencias

- [Project Gutenberg — Terms of Use](https://www.gutenberg.org/policy/terms_of_use)
- [Project Gutenberg — Permissions, Licensing and other Common Requests](https://www.gutenberg.org/policy/permission.html)
- [Project Gutenberg — License](https://www.gutenberg.org/policy/license.html)
- [Manifiesto del proyecto](manifest.csv)

**Fecha de consulta/documentación:** 24 de noviembre de 2026.
