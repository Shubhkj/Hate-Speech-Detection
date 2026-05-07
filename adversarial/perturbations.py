import random
import string

# -----------------------------
# Leetspeak Mapping
# -----------------------------
LEET_MAP = {
    "a": "@",
    "i": "1",
    "e": "3",
    "o": "0",
    "s": "$",
    "l": "1"
}


def leetspeak(text, prob=0.4):
    new_text = ""
    for char in text:
        if char.lower() in LEET_MAP and random.random() < prob:
            new_text += LEET_MAP[char.lower()]
        else:
            new_text += char
    return new_text


# -----------------------------
# Character Repetition
# -----------------------------
def char_repeat(text, prob=0.15):
    new_text = ""
    for char in text:
        new_text += char
        if char.isalpha() and random.random() < prob:
            new_text += char  # repeat once
    return new_text


# -----------------------------
# Character Deletion
# -----------------------------
def char_delete(text, prob=0.12):
    return "".join(
        char for char in text
        if not (char.isalpha() and random.random() < prob)
    )


# -----------------------------
# Character Insertion
# -----------------------------
def char_insert(text, prob=0.12):
    new_text = ""
    for char in text:
        new_text += char
        if char.isalpha() and random.random() < prob:
            new_text += random.choice(string.ascii_lowercase)
    return new_text


# -----------------------------
# Random Spacing
# -----------------------------
def random_spacing(text, prob=0.12):
    new_text = ""
    for char in text:
        new_text += char
        if char.isalpha() and random.random() < prob:
            new_text += " "
    return new_text


# -----------------------------
# Combined Attack
# -----------------------------
def apply_random_attack(text):
    attack_functions = [
        leetspeak,
        char_repeat,
        char_delete,
        char_insert,
        random_spacing
    ]
    attack = random.choice(attack_functions)
    return attack(text)