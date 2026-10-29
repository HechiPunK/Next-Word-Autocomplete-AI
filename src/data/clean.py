import re

START_RE = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
HEADING_RE = re.compile(r"^\s*(chapter|book|part|volume|contents)\b.*$", re.I | re.M)


def strip_gutenberg(text: str) -> str:
    """Quita la licencia de Project Gutenberg al inicio y al final."""
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    e = END_RE.search(text)
    if e:
        text = text[:e.start()]
    return text


def normalize(text: str) -> str:
    """Minúsculas, solo letras y apóstrofes, espacios simples."""
    text = text.replace("\ufeff", "")
    text = HEADING_RE.sub(" ", text)
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("—", " "), ("–", " "), ("_", "")):
        text = text.replace(a, b)
    text = text.lower()
    text = re.sub(r"[^a-z' \n]", " ", text)             # deja letras y apóstrofes
    text = re.sub(r"'(?![a-z])|(?<![a-z])'", " ", text)  # quita comillas sueltas
    return re.sub(r"\s+", " ", text).strip()


def clean_book(raw_text: str) -> str:
    return normalize(strip_gutenberg(raw_text))