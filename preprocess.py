import html
import re


def clean_text(text):
    """Normalize a raw review for vectorization."""
    # Replace HTML line breaks and any other tags with spaces
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities such as &amp; and &quot;
    text = html.unescape(text)
    text = text.lower()
    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text