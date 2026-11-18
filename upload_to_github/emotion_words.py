"""Word / phrase lexicons used to spot "interesting" reaction moments.

A moment is considered emotional when a streamer says something from these
lists.  We keep three categories because the user explicitly wants clips to
start with admiration, delight or aggression.  Scores are weights: 3 is a very
strong signal, 1 is weak.  Both Russian and English are supported.

This is a deliberate, fully local alternative to an LLM: no API key, no limits,
works offline once the Whisper model is cached.
"""

from __future__ import annotations

from typing import Dict, Tuple

# category -> {phrase: weight}
# Phrases may be one word or a short n-gram (up to 3 words).
LEXICON: Dict[str, Dict[str, int]] = {
    "admiration": {
        "wow": 3,
        "омг": 3,
        "omg": 3,
        "невероятно": 3,
        "incredible": 3,
        "unbelievable": 3,
        "insane": 3,
        "ахуеть": 3,
        "ахуею": 3,
        "боже": 2,
        "бог мой": 3,
        "oh my god": 3,
        "oh my": 2,
        "красота": 2,
        "красиво": 2,
        "великолепно": 2,
        "амейзинг": 3,
        "amazing": 3,
        "epic": 2,
        "эпично": 2,
        "легендарно": 3,
        "legendary": 3,
        "перебор": 1,
        "overpowered": 2,
        "гениально": 3,
        "гений": 2,
        "genius": 3,
        "браво": 2,
        "bravo": 2,
        "шок": 2,
        "shocked": 2,
        "вау": 3,
        "вауу": 3,
        "ого": 2,
        "огонь": 2,
        "this is crazy": 3,
        "let's go": 3,
        "lets go": 3,
        "yeah boy": 2,
        "погнали": 2,
        "давай давай": 2,
    },
    "delight": {
        "хаха": 2,
        "хахаха": 3,
        "лол": 2,
        "lol": 2,
        "lmao": 3,
        "haha": 2,
        "смеюсь": 2,
        "смех": 2,
        "кайф": 2,
        "угар": 2,
        "прикол": 2,
        "фан": 2,
        "fun": 2,
        "funny": 2,
        "весело": 2,
        "радость": 2,
        "счастлив": 2,
        "обожаю": 2,
        "love it": 3,
        "идеально": 2,
        "perfect": 2,
        "класс": 2,
        "классно": 2,
        "супер": 2,
        "great": 2,
        "awesome": 3,
        "обалдеть": 2,
        "забавно": 2,
        "ура": 2,
        "успех": 2,
        "победа": 3,
        "win": 2,
        "победил": 3,
        "выиграл": 3,
    },
    "aggression": {
        "бля": 3,
        "блять": 3,
        "сука": 3,
        "ёбан": 3,
        "ебан": 3,
        "пиздец": 3,
        "ахуй": 3,
        "чёрт": 2,
        "черт": 2,
        "даёшь": 1,
        "fuck": 3,
        "fucking": 3,
        "shit": 3,
        "damn": 2,
        "wtf": 3,
        "what the fuck": 3,
        "rage": 3,
        "злость": 3,
        "злюсь": 3,
        "бесит": 3,
        "бесит меня": 3,
        "ненавижу": 3,
        "hate": 3,
        "убийца": 2,
        "убить": 2,
        " damn": 2,
        " asshole": 3,
        "bastard": 3,
        "destroy": 2,
        "уничтожить": 2,
        "не может быть": 2,
        "невозможно": 2,
        "impossible": 2,
        "опять блять": 3,
        "почему блять": 3,
        "why the hell": 3,
        "seriously": 1,
        "заебал": 3,
        "заебало": 3,
        "пиздить": 2,
        "драться": 2,
    },
}

# Flat lookup: phrase -> (category, weight)
FLAT: Dict[str, Tuple[str, int]] = {}
for _category, _phrases in LEXICON.items():
    for _phrase, _weight in _phrases.items():
        FLAT[_phrase.strip().lower()] = (_category, _weight)

# All unique words that appear anywhere (for fast pre-filtering).
ALL_WORDS = {word for phrase in FLAT for word in phrase.split()}
