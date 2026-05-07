# preprocessing/clean_text.py

import re
import string

URL_PATTERN = re.compile(r"http\S+|www\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def normalize_obfuscations(text: str) -> str:
    """
    Normalize common obfuscated profanity and hate words
    using character-level regex (robust to symbols and missing letters).
    """
    patterns = [
        (r"b[^a-z]*i?[^a-z]*t[^a-z]*c[^a-z]*h", "bitch"),
        (r"f[^a-z]*u?[^a-z]*c[^a-z]*k", "fuck"),
        (r"s[^a-z]*h[^a-z]*i?[^a-z]*t", "shit"),
        (r"a[^a-z]*s[^a-z]*s", "ass"),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def clean_text(text: str) -> str:
    """
    Clean and normalize input text for hate speech detection.
    """

    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = URL_PATTERN.sub("", text)

    # 3. Normalize obfuscated words FIRST (CRITICAL)
    text = normalize_obfuscations(text)

    # 4. Remove mentions
    text = MENTION_PATTERN.sub("", text)

    # 5. Remove hashtag symbol but keep word
    text = HASHTAG_PATTERN.sub("", text)

    # 6. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 7. Normalize whitespace
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text


# -------------------------------
# Standalone test
# -------------------------------
if __name__ == "__main__":
    sample_texts = [
        "You are a b!tch!!!",
        "Go to hell f@ck you",
        "Visit http://example.com now",
        "@user This is #hate speech"
    ]

    for t in sample_texts:
        print(f"Original: {t}")
        print(f"Cleaned : {clean_text(t)}\n")
