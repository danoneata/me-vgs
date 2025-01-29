LANG_SHORT_TO_LONG = {
    "en": "english",
    "fr": "french",
    "nl": "dutch",
}

LANG_LONG_TO_SHORT = {v: k for k, v in LANG_SHORT_TO_LONG.items()}

LANGS_SHORT = list(sorted(LANG_SHORT_TO_LONG.keys()))
LANGS_LONG = list(sorted(LANG_LONG_TO_SHORT.keys()))