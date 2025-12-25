# app/main.py
import base64
import os
from fastapi import FastAPI, Depends, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from . import models, schemas
from sqlalchemy.orm import Session, joinedload, load_only # <--- Upewnij się, że są joinedload i load_only
from . import models, schemas
SongModel = models.Song
TagModel = models.Tag
SongSchema = schemas.SongMasterBase # Używamy nowego schematu

import torch
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
import pandas as pd

import spacy
from .database import SessionLocal, engine
from typing import List, Optional
from sqlalchemy import or_

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

import torch
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from collections import defaultdict
from math import log
from numpy.linalg import norm
import re
from gliner import GLiNER
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from langdetect import detect, DetectorFactory
# Jeżeli chcesz tworzyć tabele z modeli (tylko gdy nie masz już tabeli)
#models.Base.metadata.create_all(bind=engine)
load_dotenv()

app = FastAPI(title="Songs API")

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "df_full_with_embeddings.parquet"
)

# DODAJ TĘ SEKCJE:
PARQUET_TAGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "df_unique_tag_embeddings.parquet"
)




df_songs = pd.read_parquet(PARQUET_PATH)
df_tag_embeddings = pd.read_parquet(PARQUET_TAGS_PATH)
##Skrypty Klaudii

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_e5 = SentenceTransformer('intfloat/multilingual-e5-base', device=device)
model_gliner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
nlp_pl = spacy.load("pl_core_news_lg")
nlp_en = spacy.load("en_core_web_md")


def _ensure_list(x):
    if isinstance(x, list): return x
    if pd.isna(x) or x is None: return []
    return [t.strip() for t in str(x).split(",") if t.strip()]

df_songs["tags_list"] = df_songs["tags"].apply(_ensure_list)
df_songs["tag_count"] = df_songs["tags_list"].apply(len)
df_songs.head(5)






# reguły: subgatunek -> nadrzędny
ALSO_ADD_PARENT = {
    # rock
    "progressive rock": "rock",
    "classic rock": "rock",
    "indie rock": "rock",
    "hard rock": "rock",
    "pop rock": "rock",
    "psychedelic rock": "rock",
    "punk rock": "rock",
    "blues rock": "rock",
    "post rock": "rock",

    # metal
    "heavy metal": "metal",
    "death metal": "metal",
    "black metal": "metal",
    "doom metal": "metal",
    "thrash metal": "metal",
    "melodic death metal": "metal",
    "symphonic metal": "metal",
    "gothic metal": "metal",
    "nu metal": "metal",
    "progressive metal": "metal",
    "power metal": "metal",
    "metalcore": "metal",

    # pop / electronic
    "indie pop": "pop",
    "synthpop": "pop",
    "drum and bass": "electronic",
}

# 2) reguły: słowa-klucze → nadrzędny
PARENT_KEYWORDS = {
    "rock": ["rock"],
    "metal": ["metal"],
    "pop": ["pop", "britpop"],
    "hip hop": ["hip hop", "rap"],
    "electronic": ["electronic", "techno", "house", "trance", "idm", "downtempo", "electro", "ambient"],
    "jazz": ["jazz"],
    "classical": ["classical"],
    "punk": ["punk"],
    "folk": ["folk"],
    "blues": ["blues"],
    "country": ["country"],
    "reggae": ["reggae"],
}

def expand_tags(tag_list):
    if tag_list is None or (isinstance(tag_list, float) and pd.isna(tag_list)):
        tag_list = []

    # normalizacja minimalna: małe litery, zamiana _ → spacja
    tags = {str(t).lower().replace("_", " ").strip() for t in tag_list}

    # 1) subgatunek → nadrzędny
    for child, parent in ALSO_ADD_PARENT.items():
        if child in tags:
            tags.add(parent)

    # 2) słowa kluczowe → nadrzędny
    for parent, kws in PARENT_KEYWORDS.items():
        if any(kw in t for t in tags for kw in kws):
            tags.add(parent)

    return sorted(tags)

# zastosowanie
df_songs["tags_list"] = df_songs["tags_list"].apply(expand_tags)



df_songs["tags_list"] = df_songs["tags_list"].apply(
    lambda lst: [tag.replace("_", " ") for tag in lst]
)



df_songs = df_songs.reset_index(drop=True)

def build_inverted_index(df_songs: pd.DataFrame, tags_col: str = "tags_list"):
    inv = defaultdict(list)
    df_count = defaultdict(int)  # document frequency: w ilu utworach wystąpił tag

    for i, tags in enumerate(df_songs[tags_col]):
        if not tags:
            continue
        seen = set()
        for t in tags:
            inv[t].append(i)
            if t not in seen:
                df_count[t] += 1
                seen.add(t)

    # zamiana listy na np
    for t in inv:
        inv[t] = np.asarray(inv[t], dtype=np.int32)

    return inv, df_count

INV_INDEX, DF_COUNT = build_inverted_index(df_songs, tags_col="tags_list")
N_SONGS = len(df_songs)
AVG_TAG_LEN = float(df_songs["tag_count"].mean()) if "tag_count" in df_songs else 10.0


#Konfig

RETRIEVAL_CONFIG = {
    "n_candidates": 200,    # początkowe top najlepiej dopasowanych wg FAISS
}

SCORE_TIERS_CONFIG = {
    "t_high": 0.9,
    "t_mid": 0.7,
    "min_final": 100,         # chcemy finalnie tyle
    "max_c_from_low_tier": 15,  # ile max brać z Tier C gdy brakuje
}

POPULARITY_CONFIG = {
    "p_high": 70,
    "p_mid": 35,
    # procentowy miks w finalnym secie (docelowy)
    "mix": {
        "high": 0.4,     # popularne
        "mid": 0.35,     # średnie
        "low": 0.25,     # niszowe
    },
    "forced_popular": 2,    # ile utworów bardzo popularnych wstawić na sztywno
    "forced_popular_min": 80,
}

SAMPLING_CONFIG = {
    "final_n": 15,
    "alpha": 2.0,   # jak mocno faworyzujemy wyższy score przy losowaniu
    "do_shuffle": True,
}

QUERY_TAGS_CONFIG = {
    "rel_keep": 0.98,   # próg względny: bierzemy tagi >= 70% najlepszego
    "abs_keep": 0.9,  # próg absolutny: odetnij totalny szum
    "min_keep": 1,     # min liczba tagów w profilu
    "max_keep": 12,    # max liczba tagów w profilu
    "top_m_per_ngram": 2,
    "min_ngram_sim": 0.7,
    "min_unigram_sim": 0.5,
    "power": 1,
    "include_unigrams": True,
}

TAG_SCORING_CONFIG = {
    "use_idf": False,           # waż tagi rzadkie wyżej
    "k1": 1.2,                 # siła normalizacji „długości dokumentu” (liczby tagów utworu)
    "b": 0.75,
    "len_norm": False,
    "query_pow": 1.0,          # możesz podnieść np. 1.2 by ostrzej różnicować wagi tagów z promptu
    "normalize_by_tags": False # alternatywna, prostsza normalizacja: score / sqrt(n_tags_utworu)
}



# Ustawiamy ziarno dla detekcji języka (żeby wyniki były powtarzalne dla krótkich tekstów)
DetectorFactory.seed = 0


# spaCy: Model do parsowania gramatycznego i Matchera (zapewnia precyzyjne wzorce)
nlp_pl = spacy.load("pl_core_news_lg")
nlp_en = spacy.load("en_core_web_md")

GENERIC_LEMMAS = [
    # Polski
    "muzyka", "utwór", "piosenka", "kawałek", "playlista", "lista", "numer", "rok", "klimat", "styl",
    # Angielski
    "music", "song", "track", "playlist", "list", "number", "vibe", "tune", "genre", "style"
]

GENERIC_VERBS = [
    # --- POLSKI (Bezokoliczniki / Lematy) ---
    # Szukanie / Chcenie
    "szukać", "poszukiwać", "chcieć", "pragnąć", "potrzebować", "woleć", "wymagać",
    # Bycie / Posiadanie
    "być", "mieć", "znajdować", "znaleźć",
    # Słuchanie / Odtwarzanie
    "słuchać", "posłuchać", "usłyszeć", "grać", "zagrać", "puszczać", "puścić", "odtworzyć", "zapodać",
    # Prośby / Rekomendacje
    "prosić", "polecić", "polecać", "rekomendować", "sugerować", "zaproponować", "dawać", "dać",

    # --- ANGIELSKI (Base forms) ---
    # Searching / Wanting
    "search", "look", "find", "want", "need", "desire", "wish", "require",
    # Being / Having
    "be", "have", "get",
    # Listening / Playing
    "listen", "hear", "play", "replay", "stream",
    # Requests
    "give", "recommend", "suggest", "show", "provide",
]

NEGATION_TERMS = [
    # PL
    "nie", "bez", "mało", "zero", "ani", "żaden", "brak", "mniej",
    # EN
    "no", "not", "without", "less", "non", "neither", "nor", "lack", "zero"
]





def create_matcher_for_nlp(nlp_instance):
    """Tworzy obiekt Matcher przypisany do konkretnego modelu językowego"""
    matcher = Matcher(nlp_instance.vocab)

    # Warunek wykluczający
    # To musi być Rzeczownik, ale nie może być na liście generycznej
    noun_filter = {
        "POS": {"IN": ["NOUN", "PROPN"]},
        "IS_STOP": False,
        "LEMMA": {"NOT_IN": GENERIC_LEMMAS}
    }

    matcher.add("FRAZA", [
        # 1. Samodzielny rzeczownik lub nazwa własna
        # Wyłapuje pojedyncze słowa kluczowe, np. "rock", "jazz", "Metallica".
        [noun_filter],

        # 2. Przymiotnik + Rzeczownik
        # Klasyczna fraza opisująca cechę obiektu, np. "szybki bas", "ciężkie brzmienie".
        [{"POS": "ADJ"}, noun_filter],

        # 3. Wyrażenie przyimkowe (opcjonalnie poprzedzone przysłówkiem)
        # Określa przeznaczenie, styl lub pochodzenie, np. "do tańca", "z klimatem", "prosto z serca".
        [{"POS": "ADV", "OP": "?"}, {"POS": "ADP"}, noun_filter],

        # 4. Przysłówek + Przymiotnik
        # Służy do wzmocnienia lub doprecyzowania cechy, np. "bardzo wesoła", "niezwykle głośny".
        [{"POS": "ADV"}, {"POS": "ADJ", "IS_STOP": False}],

        # 5. Samodzielny przymiotnik
        # Gdy użytkownik używa samej cechy bez rzeczownika (częste w mowie potocznej), np. "rockowa", "spokojne".
        [{"POS": "ADJ", "IS_STOP": False}],

        # 6. Rzeczowniki złożone (Dwa rzeczowniki obok siebie)
        # Wyłapuje gatunki lub nazwy dwuczłonowe, np. "post rock", "hip hop", "death metal".
        [{"POS": {"IN": ["NOUN", "PROPN"]}, "IS_STOP": False}, noun_filter],

        # 7. Rozbudowana fraza przymiotnikowa z przyimkiem
        # Opisuje przydatność lub relację, np. "dobra do tańca", "idealny na imprezę".
        [{"POS": "ADV", "OP": "?"}, {"POS": "ADJ"}, {"POS": "ADP"}, noun_filter],

        # 8. Fraza czasownikowa (Czasownik + Dopełnienie)
        # Wyklucza ogólne czasowniki
        [
            {"POS": "VERB", "LEMMA": {"NOT_IN": GENERIC_VERBS}},
            {"POS": {"IN": ["NOUN", "ADJ", "PRON"]}, "OP": "+"} # Dopełnienie (może składać się z kilku słów)
        ]
    ])
    return matcher

# Tworzymy matchery raz na starcie
matcher_pl = create_matcher_for_nlp(nlp_pl)
matcher_en = create_matcher_for_nlp(nlp_en)





# 4. FUNKCJA POMOCNICZA: SPRAWDZANIE NEGACJI

def is_span_negated(doc, start_index, window=2):
    """
    Sprawdza, czy przed frazą (start_index) stoi słowo przeczące.
    Patrzy 'window' tokenów wstecz.
    """
    lookback = max(0, start_index - window)
    preceding_tokens = doc[lookback:start_index]

    for token in preceding_tokens:
        if token.text.lower() in NEGATION_TERMS:
            return True
    return False






# ==============================================================================
# GŁÓWNA FUNKCJA EKSTRAKCJI
# ==============================================================================

def extract_relevant_phrases(prompt):
    # Setup językowy
    prompt = prompt.lower()
    prompt_clean = prompt.strip()

    if not prompt_clean:
        return []

    try:
        lang_code = detect(prompt)
    except:
        lang_code = 'pl'

    if lang_code == 'en':
        current_nlp = nlp_en
        current_matcher = matcher_en
        lang_msg = "EN"
    else:
        current_nlp = nlp_pl
        current_matcher = matcher_pl
        lang_msg = "PL"

    doc = current_nlp(prompt)

    # Matcher
    matcher_matches = current_matcher(doc)

    # Konwersja matchy na obiekty Span
    matcher_spans = [doc[start:end] for match_id, start, end in matcher_matches]

    # filter_spans usuwa nakładające się frazy (np. "rock" wewnątrz "post rock")
    # zostawiając najdłuższe dopasowanie
    combined_spans = filter_spans(matcher_spans)

    final_phrases = []

    for span in combined_spans:
        # Sprawdzanie negacji
        if is_span_negated(doc, span.start):
            continue

        final_phrases.append(span.text.lower())

    # Czyszczenie i sortowanie wyników
    unique_phrases = sorted(list(set([p.strip() for p in final_phrases if len(p.strip()) > 2])))

    print(f"[{lang_msg}] Prompt: '{prompt}' \n-> {unique_phrases}")

    return unique_phrases



# ----------------------------------------------------------------------
# PRZYKŁADY UŻYCIA
# ----------------------------------------------------------------------

TEST_PROMPT_1 = "szukam muzyki rockowej, ale takiej pełnej spokoju i bardzo wesołej, trochę do tańca"
TEST_PROMPT_2 = "rock, pop, dance i coś do tańca, zależy mi na maksymalnej energii"
TEST_PROMPT_3 = "rock, pop, dance i coś do tańca, zależy mi na maksymalnej energii, post rock, alternative rock, rock alternatywny, pop"
TEST_PROMPT_4 = "muzyka rockowa z lat 90, z klimatem podróży, postpankowa"
TEST_PROMPT_5 = "muzyka bez słów, smutna, nostalgiczna, spokojna"
TEST_PROMPT_6 = "szybka, intensywna, bardzo dobra do tańca, zabawy, zajsta do tańca, idealna do tańca, "
TEST_PROMPT_7 = "albo zwykły rock, albo jakiś post rock albo punk, coś takiego"
TEST_PROMPT_8 = "muzyka dynamiczna, szybko, szybkie tempo, wysokie tempo, energiczna"
TEST_PROMPT_9 = "muzyka, która koi nerwy"
TEST_PROMPT_10 = "muzyka, która koi nerwy"
TEST_PROMPT_11 = "Szukam muzyki rockowej, ale nie smutnej"
TEST_PROMPT_12 = "I want energetic songs, no slow music"
TEST_PROMPT_13 = "Chcę wesołe piosenki, coś szybkiego, ruchliwego, rock, muzyka popowa, z lat 90, dużo gitary, shoegaze, "
# TEST_PROMPT_14 = "Vaporwave, J-Core (Japanese Hardcore), Blackened Death-Doom, Skweee, Dungeon Synth, Breakcore, Mathcore, Drone Metal (Ambient Drone), Folktronica, Lowercase"
TEST_PROMPT_14 = "szybki rock"



# extract_and_route(TEST_PROMPT_4)


# ==============================================================================
# KONFIGURACJA DO KLASYFIKACJI FRAZ
# ==============================================================================

# GLiNER ma odróżnić Tag (np. rock) od cech audio (np. szybka).

LABELS_CONFIG = {
    # --- TAGI ---
    "gatunek_muzyczny": {
        "desc": "rock, pop, jazz, hip hop, metal, indie, alternative, emo, psychedelic, industrial, grunge, punk, folk, electronic, experimental, noise music",
        "route": "TAGS"
    },
    "klimat_styl": {
        "desc": "chill, chillout, mellow, ambient, lounge, dark, beautiful, love",
        "route": "TAGS"
    },
    "typ_utworu": {
        "desc": "soundtrack, ost, muzyka filmowa, ścieżka dźwiękowa, remix",
        "route": "TAGS"
    },
    "instrument": {
        "desc": "piano, guitar, drums, violin, bass, saxophone, synthesizer, vocals",
        "route": "TAGS"
    },
    "okres_czasu": {
        "desc": "80s, 90s, 00s, 2020s, oldies, retro, klasyk, lata 90, lata 80",
        "route": "TAGS"
    },
    "pochodzenie": {
        "desc": "polish, american, british, french, k-pop, latino, spanish, deutsch",
        "route": "TAGS"
    },

    # --- WSZYSTKO INNE (dane audio) ---
    # Zamiast rozbijać na valence/tempo, wrzucamy wszystko co jest opisem tutaj.
    "cecha_audio": {
        "desc": "sad, happy, fast, slow, danceable, party, energetic, calm, relaxing, loud, quiet, acoustic, electronic, melancholic, gloomy, euphoric, club banger",
        "route": "AUDIO"
    }
}

# Generowanie zmiennych
GLINER_LABELS = [f"{k} ({v['desc']})" for k, v in LABELS_CONFIG.items()]
ROUTING_MAP = {k: v['route'] for k, v in LABELS_CONFIG.items()}




def get_label_config_lists(config):
    """
    Tworzy listę etykiet dla GLiNERa (klucz + opis) oraz mapę powrotną.
    """
    gliner_labels = []
    label_mapping = {} # Mapa "nazwa (opis)" -> "nazwa"

    for key, value in config.items():
        full_label = f"{key} ({value['desc']})"
        gliner_labels.append(full_label)
        label_mapping[full_label] = key

    return gliner_labels, label_mapping

GLINER_LABELS_LIST, GLINER_LABEL_MAP = get_label_config_lists(LABELS_CONFIG)

def classify_phrases_with_gliner(prompt, spacy_phrases, model, threshold=0.3):
    """
    Klasyfikuje frazy wykryte przez spaCy używając GLiNERa uruchomionego na całym prompcie.
    """
    if not spacy_phrases:
        return []

    # Uruchamiamy GLiNER na całym tekście (kontekst jest kluczowy)
    # Dzięki temu odróżni "Rock" (gatunek) od "Szybka" (audio)
    gliner_predictions = model.predict_entities(prompt, GLINER_LABELS_LIST, threshold=threshold)

    results = []

    # Iterujemy po frazach ze spaCy i szukamy dla nich etykiety w wynikach GLiNERa
    for phrase in spacy_phrases:
        matched_category = None
        matched_route = "AUDIO" # Domyślny routing (bezpieczny fallback)

        # Normalizacja frazy spaCy do porównania
        phrase_clean = phrase.lower().strip()

        # Szukamy czy ta fraza została też znaleziona przez GLiNERa
        # Sprawdzamy czy tekst encji GLiNERa zawiera się w frazie spaCy lub odwrotnie
        best_score = 0

        for entity in gliner_predictions:
            entity_text = entity['text'].lower().strip()

            # Sprawdzenie pokrycia (overlap)
            if phrase_clean in entity_text or entity_text in phrase_clean:
                # Jeśli mamy dopasowanie, pobieramy czystą nazwę kategorii
                full_label = entity['label']
                short_key = GLINER_LABEL_MAP.get(full_label)

                if short_key:
                    matched_category = short_key
                    # Pobieramy routing z konfiguracji
                    matched_route = LABELS_CONFIG[short_key]['route']
                    break # Znaleźliśmy etykietę, idziemy do następnej frazy spaCy

        # Jeśli GLiNER nic nie znalazł dla tej frazy, ale spaCy ją wykryło
        # oznaczamy jako ogólną "cecha_audio" (w dalszych etapach E5 może ją oznaszyć jako "śmieć")
        if not matched_category:
            matched_category = "cecha_audio" # Fallback
            matched_route = "AUDIO"

        results.append({
            "phrase": phrase,
            "category": matched_category,
            "route": matched_route
        })

    return results




def prepare_queries_for_e5_separated(classified_data, original_prompt):
    """
    Rozdziela frazy na Tagi i Audio.
    Zwraca słownik z dwoma listami: "tags_to_match" i "audio_features".
    """

    return {
        "tags_to_match": [
            item['phrase'] for item in classified_data
            if item['route'] == 'TAGS'
        ],
        "audio_features": [
            item['phrase'] for item in classified_data
            if item['route'] == 'AUDIO'
        ]
    }






FEATURE_DESCRIPTIONS = {
    'valence': [
        (0.1, "very low valence, very sad, melancholic, dark, gloomy emotional mood music"),
        (0.3, "low valence, bittersweet, thoughtful, introspective, moody emotional mood music"),
        # (0.5, "medium valence, neutral emotional mood, neither clearly happy nor clearly sad music"),
        (0.7, "high valence, positive, pleasant, warm, cheerful, uplifting emotional mood music"),
        (0.9, "very high valence, very happy, joyful, exstatic, euphoric, bright, feel-good emotional mood music")
    ],

    'danceability': [
        (0.1, "very low danceability, not danceable, abstract or experimental, weak or irregular rhythm music"),
        (0.3, "low danceability, little groove, minimal rhythm, not primarily for dancing music"),
        # (0.5, "medium danceability, some groove, steady rhythm music"),
        (0.7, "high danceability, clear beat, strong groove, good for dancing, club-oriented music"),
        (0.9, "very high danceability, strong groove, infectious rhythm, perfect for dancing, party, club banger music")
    ],

    'acousticness': [
        (0.1, "very low acousticness, fully electronic, synthetic, digital, computer-generated sound music"),
        (0.3, "low acousticness, mostly electronic with some subtle organic or acoustic elements music"),
        # (0.5, "medium acousticness, balanced mix of acoustic and electronic instruments, hybrid sound music"),
        (0.7, "high acousticness, mostly acoustic, organic, live instruments such as accoustic guitar or piano music"),
        (0.9, "very high acousticness, fully acoustic, unplugged, natural, organic instruments only music")
    ],

    'n_tempo': [
        (0.1, "very slow tempo, very slow pace, dragging rhythm music"),
        (0.3, "slow tempo, downtempo, slow pace, relaxed rhythm music"),
        (0.5, "medium tempo, moderate pace, walking pace music"),
        (0.7, "fast tempo, uptempo, quick pace, energetic rhythm music"),
        (0.9, "very fast tempo, rapid pace, racing rhythm, frantic speed music")
    ],

    'instrumentalness': [
        (0.1, "very low instrumentalness, strong presence of vocals and lyrics, clear singing, vocal-focused track"),
        (0.5, "medium instrumentalness, mix of vocals and instrumental sections, vocals present but not constant"),
        (0.9, "very high instrumentalness, fully instrumental track, no vocals, no singing, music without lyrics")
    ],

    'energy': [
        (0.1, "very low energy, motionless, static, sleep-inducing, minimal activity music"),
        (0.3, "low energy, relaxed, laid-back, mellow, slow-moving atmosphere music"),
        (0.5, "medium energy, moderate pace, steady rhythm, balanced activity music"),
        (0.7, "high energy, active, driving rhythm, fast-paced, stimulating, busy arrangement music"),
        (0.9, "very high energy, hyperactive, restless, frantic, adrenaline-pumping, non-stop action music")
    ],

    'n_loudness': [
        (0.1, "very low loudness, barely audible, near silence, whisper-like volume, extremely quiet music"),
        (0.3, "low loudness, soft volume, background level, reduced amplitude, delicate sound music"),
        (0.5, "medium loudness, standard volume, normal mastering level music"),
        (0.7, "high loudness, loud volume, amplified sound, noisy, high amplitude music"),
        (0.9, "very high loudness, maximum volume, deafening, high decibels music")
    ],

    'speechiness': [
        (0.1, "very low speechiness, purely musical track, no spoken words, fully melodic music"),
        (0.3, "low speechiness, mostly music with occasional spoken words or short background phrases"),
        (0.5, "medium speechiness, balanced mix of speech and music, frequent spoken segments, rap-like or talky structure"),
    ],

    'noise': [
        # Rzeczowniki (generyczne słowa)
        (None, "music song track playlist list recording audio sound genre style vibe type kind number piece"),
        # TODO: przetestować, czy to się nie będzie myliło z instrumentallness

        # Czasowniki (związane z szukaniem)
        (None, "I am looking for I want I need search find play listen to give me recommend show me"),

        # Przymiotniki (fillery bez konkretnej treści)
        (None, "good very good nice great best cool amazing awesome some any kind of such a")
    ]
}



def prepare_feature_embeddings():
    """Embeduj wszystkie opisy jako 'passages' (raz przy starcie)"""

    feature_embeddings = {}

    for feature_name, descriptions in FEATURE_DESCRIPTIONS.items():
        # E5 wymaga "passage: " dla dokumentów
        passages = [f"passage: {desc}" for value, desc in descriptions]
        embeddings = model_e5.encode(passages, normalize_embeddings=True)
        feature_embeddings[feature_name] = embeddings

    return feature_embeddings

# Setup (raz)
feature_embeddings = prepare_feature_embeddings()




def phrases_to_features(phrases_list, confidence_threshold=0.78, lang_code='pl'):
    """
    Logika:
    1. Sprawdź surową frazę (bez "muzyka") ze wszystkimi kategoriami (w tym Noise).
       Jeśli wygra Noise -> odrzuć.
    2. Dla pozostałych dodaj "muzyka", aby precyzyjnie dopasować cechę.
    """
    if not phrases_list:
        return []

    # ETAP 1: ELIMINACJA (Surowe frazy vs Wszystkie kategorie)
    # --------------------------------------------------------

    # Kodujemy frazy BEZ słowa "muzyka"
    raw_queries = [f"query: {p}" for p in phrases_list]
    raw_embeddings = model_e5.encode(raw_queries, normalize_embeddings=True)

    valid_indices = [] # Indeksy fraz, które przeszły selekcję

    for i, phrase in enumerate(phrases_list):
        current_raw_emb = raw_embeddings[i].reshape(1, -1)

        # Szukamy zwycięzcy dla surowej frazy
        best_cat_raw = None
        best_score_raw = -1.0

        for feature_name, _ in FEATURE_DESCRIPTIONS.items():
            target_embs = feature_embeddings[feature_name]

            # Porównanie
            sims = cosine_similarity(current_raw_emb, target_embs)[0]
            max_score = float(np.max(sims))

            if max_score > best_score_raw:
                best_score_raw = max_score
                best_cat_raw = feature_name

        # DECYZJA: Czy wygrał Noise?
        if best_cat_raw == 'noise':
            print(f" Odrzucono (wygrał Noise): '{phrase}' (score: {best_score_raw:.3f})")
            continue # Fraza odpada
        else:
            # Fraza przeszła, zapamiętujemy jej indeks
            valid_indices.append(i)

    if not valid_indices:
        return []

    # ETAP 2: PRECYZYJNE DOPASOWANIE (Z kontekstem "Muzyka")
    # ------------------------------------------------------

    # Wyciągamy frazy, które przetrwały
    clean_phrases = [phrases_list[i] for i in valid_indices]

    # Dodajemy kontekst (suffix)
    suffix = " muzyka" if lang_code == 'pl' else " music"
    context_queries = [f"query: {suffix} {p}" for p in clean_phrases]
    context_embeddings = model_e5.encode(context_queries, normalize_embeddings=True)

    found_features = {}

    for i, phrase in enumerate(clean_phrases):
        current_emb = context_embeddings[i].reshape(1, -1)

        phrase_best_cat = None
        phrase_best_val = None
        phrase_best_score = -1.0
        phrase_best_desc = ""

        # Iterujemy po cechach (już BEZ Noise, bo go wyeliminowaliśmy)
        for feature_name, descriptions in FEATURE_DESCRIPTIONS.items():
            if feature_name == 'noise':
                continue

            target_embs = feature_embeddings[feature_name]
            sims = cosine_similarity(current_emb, target_embs)[0]

            max_idx = int(np.argmax(sims))
            max_score = float(sims[max_idx])

            if max_score > phrase_best_score:
                phrase_best_score = max_score
                phrase_best_cat = feature_name
                phrase_best_val = descriptions[max_idx][0]
                phrase_best_desc = descriptions[max_idx][1]

        # Zapisujemy wynik (jeśli przekroczył próg jakości)
        if phrase_best_score >= confidence_threshold:
            print(f"   Fraza: '{phrase}' -> {phrase_best_cat} ({phrase_best_score:.3f})")

            # Rozwiązywanie konfliktów (Max Pooling)
            if phrase_best_cat in found_features:
                if phrase_best_score > found_features[phrase_best_cat]['confidence']:
                    found_features[phrase_best_cat] = {
                        'value': phrase_best_val,
                        'confidence': phrase_best_score,
                        # 'matched_description': phrase_best_desc
                    }
            else:
                found_features[phrase_best_cat] = {
                    'value': phrase_best_val,
                    'confidence': phrase_best_score,
                    # 'matched_description': phrase_best_desc
                }

    sorted_features = sorted(
        found_features.items(),
        key=lambda x: x[1]['confidence'],
        reverse=True
    )

    return sorted_features











df_tag_embeddings

TAG_VECS = np.array(df_tag_embeddings["tag_embedding"].to_list(), dtype=np.float32)
TAGS = df_tag_embeddings["tag"].tolist()



def map_phrases_to_tags(
    phrases: list[str],
    model,
    tag_vecs,
    tags_list,
    threshold: float = 0.65
) -> dict[str, float]:
    """
    Mapuje listę fraz z zapytania użytkownika na konkretne tagi z bazy danych.
    Stosuje strategię "Winner Takes All" (1 Fraza = 1 Najlepszy Tag).

    Dla każdej frazy (np. "szybki rock") oblicza podobieństwo do wszystkich dostępnych tagów
    i wybiera ten jedyny, który pasuje najlepiej (np. "Rock"), o ile przekracza próg pewności.

    Args:
        phrases (list[str]): Lista fraz wyekstrahowanych z promptu (np. przez GLiNER/SpaCy).
                             Przykład: ['rock', 'lata 90', 'gitara'].
        model: Model embeddingów (np. E5/SentenceTransformer), posiadający metodę `.encode()`.
               Musi być tym samym modelem, którym zwektoryzowano tagi.
        tag_vecs (np.ndarray): Macierz pre-kalkulowanych embeddingów tagów.
                               Wymiary: (Liczba_Tagów x Wymiar_Wektora).
        tags_list (list[str]): Lista nazw tagów odpowiadająca wierszom w `tag_vecs`.
                               Indeks i w `tags_list` musi odpowiadać wierszowi i w `tag_vecs`.
        threshold (float, optional): Minimalny próg podobieństwa (0.0 - 1.0).
                                     Jeśli najlepszy tag ma wynik niższy niż próg, fraza jest ignorowana.
                                     Domyślnie 0.65.

    Returns:
        dict[str, float]: Słownik znalezionych tagów i ich wag (podobieństwa).
                          Klucz: Nazwa taga (z `tags_list`).
                          Wartość: Score podobieństwa (0.0 - 1.0).
                          Jeśli kilka fraz wskazuje na ten sam tag, zachowywany jest najwyższy wynik.
    """
    if not phrases:
        return {}

    print(f"Mapowanie 1:1 dla fraz: {phrases}")

    # Batch Encoding (Szybkie wektoryzowanie zapytań)
    # Dodajemy prefix 'query:' zgodnie ze standardem E5
    q_vecs = model.encode(
        [f"query: {p}" for p in phrases],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Obliczenie macierzy podobieństwa (Tags x Phrases)
    sims_matrix = cosine_similarity(tag_vecs, q_vecs)

    found_tags = {}

    # Iteracja po każdej frazie (kolumnie macierzy)
    for i, phrase in enumerate(phrases):
        # Pobieramy kolumnę wyników dla bieżącej frazy
        phrase_scores = sims_matrix[:, i]

        # Znajdujemy indeks najlepszego taga (argmax)
        best_idx = np.argmax(phrase_scores)
        best_score = float(phrase_scores[best_idx])
        best_tag_name = tags_list[best_idx]

        # Sprawdzamy, czy wynik przekaracza threshold
        if best_score >= threshold:
            print(f"    Fraza '{phrase}' -> Tag '{best_tag_name}' ({best_score:.3f})")

            # Zapisujemy wynik.
            # Jeśli inny tag już był zapisany (np. inna fraza też wskazała na "Rock"),
            # bierzemy ten z wyższym scorem (Max Pooling)
            if best_tag_name in found_tags:
                found_tags[best_tag_name] = max(found_tags[best_tag_name], best_score)
            else:
                found_tags[best_tag_name] = best_score
        else:
            print(f"    Fraza '{phrase}' nie pasuje do żadnego tagu (max: {best_score:.3f})")
            pass

    return found_tags




def get_query_tag_weights(
    raw_tags
) -> dict[str, float]:
    """
    Normalizuje wagi tak, by sumowały się do 1.
    """
    s = sum(raw_tags.values())
    if s <= 0:
        return raw_tags

    return {t: v / s for t, v in raw_tags.items()}





def score_songs_by_tags(
    query_tag_weights: dict[str, float],
    inv_index: dict[str, np.ndarray],
    n_songs: int,
    idf_map: dict[str, float],
    use_idf: bool = True,
    query_pow: float = 1.0,
) -> np.ndarray:
    """
    Liczy dopasowanie (score) wszystkich utworów w bazie do tagów z zapytania.
    Działa w oparciu o szybki odwrócony indeks (Inverted Index).

    Algorytm sumuje wagi tagów znalezionych w piosence, opcjonalnie mnożąc je przez IDF
    (rzadkość tagu), a na końcu normalizuje wynik do zakresu 0.0 - 1.0.

    Args:
        query_tag_weights (dict[str, float]): Znormalizowane wagi tagów wyciągnięte z promptu.
            Klucz: nazwa tagu (np. 'rock'), Wartość: waga (np. 0.6).
        inv_index (dict[str, np.ndarray]): Odwrócony indeks bazy utworów.
            Klucz: nazwa tagu. Wartość: tablica indeksów (ID) piosenek, które mają ten tag.
            Umożliwia błyskawiczne wyszukiwanie bez iteracji po wszystkich piosenkach.
        song_tag_count (np.ndarray): Wektor o długości równej liczbie piosenek w bazie (`n_songs`).
            Służy tutaj do określenia wymiaru wektora wyników.
        idf_map (dict[str, float]): Mapa Inverse Document Frequency.
            Klucz: tag. Wartość: współczynnik rzadkości.
            Rzadkie tagi (np. 'harfa') mają wyższe IDF niż częste (np. 'pop'),
            co pozwala premiować unikalne dopasowania.
        use_idf (bool, optional): Czy uwzględniać rzadkość tagów przy scoringu.
            Domyślnie True. Zalecane, aby unikać faworyzowania bardzo popularnych gatunków.
        query_pow (float, optional): Wykładnik potęgi do wyostrzania wag z promptu.
            - 1.0: brak zmian (liniowo).
            - > 1.0 (np. 2.0): Zwiększa przepaść między ważnymi a mało ważnymi tagami z zapytania.
            Domyślnie 1.0.

    Returns:
        np.ndarray: Wektor o kształcie `(n_songs,)` z wartościami float w zakresie [0.0, 1.0].
            Indeks w wektorze odpowiada indeksowi piosenki w `df_songs`.
    """

    # Przetworzenie wag zapytania
    qtw = {}

    # Zmienna do trzymania maksymalnego możliwego wyniku
    max_theoretical_score = 0.0

    for t, w in query_tag_weights.items():
        w_processed = w ** query_pow

        # Pobieramy IDF dla tego taga
        this_idf = 1.0
        if use_idf and idf_map is not None:
            this_idf = float(idf_map.get(t, 1.0))

        # Zapisujemy finalną wagę punktową dla tego taga
        final_weight = w_processed * this_idf

        qtw[t] = final_weight

        # Dodajemy do mianownika: Idealna piosenka miałaby wszystkie te tagi
        max_theoretical_score += final_weight

    # Inicjalizacja wektora wyników
    scores = np.zeros(n_songs, dtype=np.float32)

    # Sumowanie punktów (Coverage)
    if max_theoretical_score > 0:
        for tag, weight in qtw.items():
            if tag not in inv_index:
                continue

            idxs = inv_index[tag]
            scores[idxs] += weight

        # normalizacja
        scores /= max_theoretical_score

    return np.clip(scores, 0.0, 1.0)


def build_idf_map(inv_index: dict[str, np.ndarray], n_songs: int) -> dict[str, float]:
    """
    Tworzy mapę IDF (Inverse Document Frequency) dla wszystkich tagów.

    Wzór: IDF = log(N / df)
    Gdzie:
      - N: całkowita liczba piosenek
      - df: liczba piosenek zawierających dany tag

    Tagi rzadkie (np. 'dark ambient') dostaną wysoki wynik (np. 3.5).
    Tagi pospolite (np. 'pop') dostaną niski wynik (np. 0.2).
    """
    idf_map = {}

    for tag, song_indices in inv_index.items():
        # df (Document Frequency) - w ilu piosenkach występuje ten tag
        df = len(song_indices)

        if df > 0:
            # Używamy logarytmu, żeby spłaszczyć wyniki
            idf_score = np.log10(n_songs / df)
            idf_map[tag] = float(idf_score)

    return idf_map


def retrieve_candidates_tags(
    scores_songs_tag: np.ndarray,
    n_candidates: int = 400,
    flat_delta: float = 0.05
):
    # bierzemy tylko te, które w ogóle coś dopasowały
    pos_mask = scores_songs_tag > 0.0
    if not np.any(pos_mask):
        return df_songs.iloc[0:0].copy()

    pos_idxs = np.where(pos_mask)[0]
    # sortujemy wszystkie dopasowane po score malejąco
    pos_idxs = pos_idxs[np.argsort(-scores_songs_tag[pos_idxs])]

    # jeśli i tak mniej niż n_candidates → bierzemy wszystkie
    if len(pos_idxs) <= n_candidates:
        final_idxs = pos_idxs
    else:
        # score na granicy top-n
        boundary_idx = n_candidates - 1
        boundary_score = float(scores_songs_tag[pos_idxs[boundary_idx]])

        # rozszerzamy cut tak długo, jak kolejne mają ten sam (w praktyce: bardzo podobny) score
        cut = n_candidates
        while (
            cut < len(pos_idxs)
            and abs(float(scores_songs_tag[pos_idxs[cut]]) - boundary_score) <= flat_delta
        ):
            cut += 1

        final_idxs = pos_idxs[:cut]

    candidates = df_songs.iloc[final_idxs].copy()
    candidates["tag_score"] = scores_songs_tag[final_idxs]
    candidates["score"] = candidates["tag_score"]  # spójność z resztą pipeline'u
    return candidates



def calculate_audio_match(candidates_df, audio_criteria):
    """
    Liczy dopasowanie kandydatów do kryteriów audio.
    Zwraca wektor (len(candidates),) z wartościami 0.0 - 1.0.
    """
    if candidates_df.empty:
        return np.array([])

    # Jeśli nie ma kryteriów audio, zwracamy 1.0 (neutralny wpływ)
    # lub 0.0 (jeśli chcemy, by audio było wymagane - ale lepiej 1.0)
    if not audio_criteria:
        return np.ones(len(candidates_df))

    # Wektor końcowy (zaczynamy od samych jedynek)
    total_audio_score = np.ones(len(candidates_df))

    # Iterujemy po każdym kryterium (np. energy=0.9, danceability=0.7)
    for feature_name, criteria in audio_criteria:
        target_val = criteria['value']  # np. 0.9

        # Sprawdzamy, czy cecha istnieje w DataFrame
        if feature_name not in candidates_df.columns:
            continue

        # Pobieramy wartości z kolumny (np. [0.4, 0.8, 0.9])
        song_values = candidates_df[feature_name].to_numpy()

        # OBLICZANIE DYSTANSU (Similarity)
        # Prosta metoda liniowa: 1.0 - dystans
        # Jeśli target=0.9, a song=0.5 -> dystans=0.4 -> score=0.6
        dist = np.abs(song_values - target_val)
        sim = 1.0 - dist

        # Opcjonalnie: Boostowanie kary za duże odchylenia (kwadrat)
        # sim = 1.0 - (dist ** 2)

        # Clip, żeby nie wyjść poza zakres (choć przy 0-1 nie powinno)
        sim = np.clip(sim, 0.0, 1.0)

        # Mnożymy (intersekcja warunków): Song musi spełniać Warunek 1 I Warunek 2
        total_audio_score *= sim

    return total_audio_score



def merge_tag_and_audio_scores(df, audio_weight=0.3, use_tags=True):
    df = df.copy()
    # Jeśli nie ma tagów, waga Audio to 100% (1.0), inaczej bierzemy z configu
    w = audio_weight if use_tags else 1.0

    # Liczymy score (safe get zabezpiecza przed brakiem kolumn)
    df['score'] = (df.get('tag_score', 0) * (1 - w)) + (df.get('audio_score', 0) * w)

    # Uzupełniamy braki zerami tylko dla estetyki wyświetlania
    for col in ['tag_score', 'audio_score']:
        if col not in df: df[col] = 0.0

    return df.sort_values('score', ascending=False)


def tier_by_score(candidates: pd.DataFrame, t_high: float, t_mid: float):
    """
    Dzieli kandydatów na trzy poziomy jakości (tiery) w zależności od wartości dopasowania (score).

    - Tier A: score >= t_high - bardzo dobre dopasowanie
    - Tier B: t_mid <= score < t_high - średnie dopasowanie
    - Tier C: score < t_mid - słabe dopasowanie

    Args:
        candidates (pd.DataFrame): Dane z kolumną 'score'.
        t_high (float): Próg dla wysokiego dopasowania.
        t_mid (float): Próg dla średniego dopasowania.

    Returns
        tuple: (tier_a, tier_b, tier_c) — trzy DataFramey z podziałem wg jakości.
    """
    tier_a = candidates[candidates["score"] >= t_high].copy()
    tier_b = candidates[(candidates["score"] < t_high) & (candidates["score"] >= t_mid)].copy()
    tier_c = candidates[candidates["score"] < t_mid].copy()
    return tier_a, tier_b, tier_c

def calculate_dynamic_thresholds(candidates_df, high_threshold=0.75, mid_threshold=0.5):
    """
    Wylicza progi t_high i t_mid na podstawie tego,
    jak dobre wyniki w ogóle udało się znaleźć.
    """
    if candidates_df.empty:
        return 0.0, 0.0

    # Jaki jest absolutnie najlepszy wynik dla tego zapytania?
    max_score = candidates_df['score'].max()

    # --- LOGIKA ADAPTACYJNA ---

    # 1. Tier A: Musi być blisko lidera (np. max - 0.1),
    #    ALE nie może być mniejszy niż 0.75 (Quality Gate).
    #    Jeśli max_score to 0.5, to t_high będzie 0.75 -> Tier A będzie pusty (i dobrze!)
    t_high = max(high_threshold, max_score - 0.1)

    # 2. Tier B: Musi być sensowny (np. max - 0.3),
    #    ALE nie mniejszy niż 0.4 (żeby nie brać śmieci).
    t_mid = max(mid_threshold, max_score - 0.2)

    return t_high, t_mid






def build_working_set(
    tier_a: pd.DataFrame,
    tier_b: pd.DataFrame,
    tier_c: pd.DataFrame,
    target_pool_size: int,
    min_required_size: int,
    popularity_rescue_ratio: float = 0.2
) -> pd.DataFrame:
    """
    Buduje zbiór roboczy.
    Ważne: Przy dobieraniu z Tier B, dba o to, aby nie wziąć tylko niszowych utworów z dobrym score,
    ale też "uratować" popularne utwory, które mają przyzwoity score (są w Tier B).
    Args:
        tier_a (pd.DataFrame): Utwory o wysokim dopasowaniu (Hity jakościowe).
                               Bierzemy je zawsze w całości.
        tier_b (pd.DataFrame): Utwory o średnim dopasowaniu (Dobre, ale nie idealne).
                               Służą do uzupełnienia puli do `target_pool_size`.
        tier_c (pd.DataFrame): Utwory o niskim dopasowaniu (Słabe).
                               Używane tylko awaryjnie, jeśli nie osiągniemy `min_required_size`.
        target_pool_size (int): Docelowa wielkość zbioru roboczego (np. 50-100).
                                Tyle kandydatów chcemy dać algorytmowi losującemu,
                                aby miał swobodę w dobieraniu High/Mid/Low popularity.
        min_required_size (int): ABSOLUTNE MINIMUM. Tyle piosenek ma mieć finalna playlista.
                                 Jeśli A+B to za mało, dobierzemy stąd resztę z Tier C.
                                 Zalecane: tyle, ile user zażądał (np. 15).
        popularity_rescue_ratio (float): Jaka część slotów z Tieru B ma być zarezerwowana
                                         dla utworów NAJPOPULARNIEJSZYCH (zamiast tych z najlepszym score).
                                         Np. 0.3 oznacza, że 30% dobieranych utworów z B to hity,
                                         a 70% to najlepiej pasujące niszowe.
    """

    # 1. ZAWSZE bierzemy całe Tier A
    working_parts = [tier_a]
    current_count = len(tier_a)

    # 2. UZUPEŁNIANIE Z TIER B
    if current_count < target_pool_size:
        needed = target_pool_size - current_count

        if len(tier_b) <= needed:
            working_parts.append(tier_b)
            current_count += len(tier_b)
        else:
            n_pop = int(needed * popularity_rescue_ratio)
            n_score = needed - n_pop

            b_pop_rescued = tier_b.sort_values("popularity", ascending=False).head(n_pop)
            remaining_b = tier_b.drop(b_pop_rescued.index)
            b_score_top = remaining_b.sort_values("score", ascending=False).head(n_score)

            working_parts.extend([b_pop_rescued, b_score_top])
            current_count += (len(b_pop_rescued) + len(b_score_top))

    # 3. RATUNEK Z TIER C
    if current_count < min_required_size:
        needed = min_required_size - current_count
        part_c = tier_c.sort_values("score", ascending=False).head(needed)
        working_parts.append(part_c)

    if not working_parts:
        return pd.DataFrame(columns=tier_a.columns)

    # 4. Sklejanie
    working = pd.concat(working_parts, ignore_index=True) # Reset indexu przy sklejaniu

    # === POPRAWKA: BEZPIECZNE USUWANIE DUPLIKATÓW ===

    # Ustalamy po czym identyfikujemy piosenkę.
    # Unikamy sprawdzania kolumn zawierających listy (jak tags_list).
    subset_cols = []

    if 'id' in working.columns:
        subset_cols = ['id']
    elif 'name' in working.columns and 'artist' in working.columns:
        subset_cols = ['name', 'artist']

    if subset_cols:
        working = working.drop_duplicates(subset=subset_cols)
    else:
        # Ostateczność: Jeśli nie mamy ID ani Nazwy, a mamy unikalny index w piosenkach
        # to po prostu nie robimy drop_duplicates() po zawartości,
        # bo concat z ignore_index=True i tak stworzył nowy indeks.
        # Ewentualnie, jeśli masz pewność, że w indexach źródłowych (tier_a/b/c)
        # były ID piosenek, to nie używaj ignore_index=True w concat.
        pass

    # Finalne sortowanie
    working = working.sort_values("score", ascending=False).reset_index(drop=True)

    return working




# working




def bucket_by_popularity(working: pd.DataFrame, p_high: int, p_mid: int):
    """
    Dzieli utwory na trzy grupy (bucket'y) według ich popularności.

    - High:  popularity >= p_high
    - Mid:   p_mid <= popularity < p_high
    - Low:   popularity < p_mid

    Args:
        working (pd.DataFrame): Zbiór roboczy utworów z kolumną 'popularity'.
        p_high (int): Próg dla wysokiej popularności.
        p_mid (int): Próg dla średniej popularności.

    Returns:
        tuple: (pop_high, pop_mid, pop_low) — trzy DataFrame'y z podziałem wg popularności.
    """

    pop_high = working[working["popularity"] >= p_high].copy()
    pop_mid = working[(working["popularity"] < p_high) & (working["popularity"] >= p_mid)].copy()
    pop_low = working[working["popularity"] < p_mid].copy()
    return pop_high, pop_mid, pop_low




def weighted_sample(df: pd.DataFrame, k: int, alpha: float):
    """
    Losuje do k wierszy z df bez zwracania, z wagami ~ score^alpha.
    Jak df ma mniej niż k wierszy → zwraca wszystkie.
    """
    if k <= 0 or len(df) == 0:
        return df.iloc[0:0]

    k = min(k, len(df))

    scores = df["score"].to_numpy(dtype=np.float32)
    scores = np.clip(scores, 1e-9, None)
    w = np.power(scores, alpha)
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.full_like(w, 1.0 / len(w), dtype=np.float32)
    else:
        w /= w_sum

    idx = np.random.choice(len(df), size=k, replace=False, p=w)
    return df.iloc[idx]
def sample_final_songs(
    working: pd.DataFrame,
    popularity_cfg: dict,
    sampling_cfg: dict,
) -> pd.DataFrame:
    """
    Wybiera finalną playlistę z puli roboczej (working set).

    Logika:
    1. Dzieli utwory na High/Mid/Low Popularity.
    2. "Forced Popular": Wybiera N hitów (jeśli skonfigurowano),
       ale losuje je ważonym scorem, żeby nie grać w kółko tego samego.
    3. Resztę playlisty dobiera z bucketów High/Mid/Low wg zadanego miksu (np. 40%/40%/20%).

    Dzięki temu, że 'score' zawiera już ocenę Audio, losowanie naturalnie
    preferuje utwory pasujące brzmieniowo.
    """

    if len(working) == 0:
        return working.iloc[0:0].copy()

    # Konfiguracja
    final_n = sampling_cfg.get("final_n", 15)
    alpha   = sampling_cfg.get("alpha", 2.0) # Siła wpływu score na losowanie

    p_high = popularity_cfg.get("p_high", 70)
    p_mid  = popularity_cfg.get("p_mid", 35)
    mix    = popularity_cfg.get("mix", {"high": 0.4, "mid": 0.35, "low": 0.25})

    forced_popular         = popularity_cfg.get("forced_popular", 0)
    forced_popular_min     = popularity_cfg.get("forced_popular_min", p_high)

    # 1) Bucketowanie po popularności (korzystamy z Twojej funkcji pomocniczej)
    pop_high, pop_mid, pop_low = bucket_by_popularity(working, p_high=p_high, p_mid=p_mid)

    final_parts = []

    # 2) FORCED POPULAR (Zmienione: Losowość zamiast sztywnego .head())
    # Wybieramy pulę bardzo popularnych utworów (powyżej forced_popular_min)
    # Zazwyczaj to podzbiór pop_high
    forced_pool = working[working["popularity"] >= forced_popular_min].copy()

    # Zamiast brać top N na sztywno, losujemy z nich, preferując te z wyższym score.
    # Dzięki temu playlista jest bardziej różnorodna przy kolejnych wywołaniach.
    forced_taken = weighted_sample(forced_pool, forced_popular, alpha)

    final_parts.append(forced_taken)

    # Usuwamy wybrane utwory z bucketów, żeby ich nie dublować
    used_idx = set(forced_taken.index)
    pop_high = pop_high[~pop_high.index.isin(used_idx)]
    pop_mid  = pop_mid[~pop_mid.index.isin(used_idx)]
    pop_low  = pop_low[~pop_low.index.isin(used_idx)]

    # 3) Obliczamy ile slotów zostało do wypełnienia
    n_forced = len(forced_taken)
    remaining = max(0, final_n - n_forced)

    if remaining == 0:
        combined = pd.concat(final_parts, ignore_index=False)
        return combined.sort_values("score", ascending=False).reset_index(drop=True)

    # 4) Wyliczamy cele dla bucketów (Mix)
    target_high = int(round(remaining * mix.get("high", 0.0)))
    target_mid  = int(round(remaining * mix.get("mid",  0.0)))
    target_low  = remaining - target_high - target_mid  # Reszta do low

    # Clamp (nie możemy wziąć więcej niż jest w buckecie)
    target_high = min(target_high, len(pop_high))
    target_mid  = min(target_mid, len(pop_mid))
    target_low  = min(target_low, len(pop_low))

    # 5) Główne losowanie z bucketów (Weighted Sample wg Score)
    # Tu dzieje się magia Audio/Tagów: piosenki, które lepiej pasują (wyższy score),
    # mają większą szansę na wylosowanie.
    sampled_high = weighted_sample(pop_high, target_high, alpha)
    sampled_mid  = weighted_sample(pop_mid,  target_mid,  alpha)
    sampled_low  = weighted_sample(pop_low,  target_low,  alpha)

    final_parts.extend([sampled_high, sampled_mid, sampled_low])

    combined = pd.concat(final_parts, ignore_index=False)

    # 6) Fill Gaps (Jeśli zaokrąglenia albo braki w bucketach sprawiły, że mamy za mało)
    if len(combined) < final_n:
        missing = final_n - len(combined)
        used_idx = set(combined.index)

        # Bierzemy resztki z całego working setu
        remaining_pool = working[~working.index.isin(used_idx)]
        extra = weighted_sample(remaining_pool, missing, alpha)

        combined = pd.concat([combined, extra], ignore_index=False)

    # 6) FINALNE TASOWANIE
    # frac=1 oznacza "weź 100% wierszy, ale w losowej kolejności"
    do_shuffle = sampling_cfg.get("shuffle", True)
    if do_shuffle:
        combined = combined.sample(frac=1).reset_index(drop=True)
    else:
        # Opcja dla debugowania: najlepsze na górze
        combined = combined.sort_values("score", ascending=False).reset_index(drop=True)

    return combined


from fastapi import FastAPI, Depends, HTTPException, Query  # Upewnij się, że masz Query
from typing import List
import numpy as np




@app.get("/search/{text}", response_model=List[schemas.SongResult])
def search_by_text(
        # Zmieniono nazwę endpointu na bardziej standardową
        text: str = Path(..., min_length=1, max_length=500), # Używamy Path, a nie Query!
        ilosc: int = Query(SAMPLING_CONFIG['final_n'], alias="top_n", ge=1, le=50)
):
    """
    Generuje listę ID utworów Spotify na podstawie promptu.
    Wykonuje cały pipeline rekomendacyjny na danych w DataFrame.
    """

    global df_songs, INV_INDEX, TAG_VECS, TAGS, feature_embeddings, model_gliner, model_e5, DF_COUNT

    # 1. WERYFIKACJA DANYCH
    if df_songs.empty or INV_INDEX is None or TAG_VECS is None or TAG_VECS.size == 0:
        raise HTTPException(status_code=500,
                            detail="Błąd: Zasoby (DataFrame, Indeksy) nie zostały załadowane poprawnie.")

    # UŻYWAMY PROMPTU PODANEGO PRZEZ UŻYTKOWNIKA (a nie stałej TEST_PROMPT_3)
    prompt =  text

    # 2. FAZA 1: EKSTRAKCJA I KLASYFIKACJA FRAZ
    # ------------------------------------------------------------------
    try:
        lang_code = detect(prompt)
    except:
        lang_code = 'pl'

    relevant_phrases = extract_relevant_phrases(prompt)

    if not relevant_phrases:
        raise HTTPException(status_code=404, detail="Brak fraz kluczowych. Zapytanie jest zbyt ogólne.")

    classified_phrases = classify_phrases_with_gliner(prompt, relevant_phrases, model_gliner)
    e5_queries_separated = prepare_queries_for_e5_separated(classified_phrases, prompt)

    tags_queries = e5_queries_separated['tags_to_match']
    audio_features_queries = e5_queries_separated['audio_features']
    # 3. FAZA 2: WYSZUKIWANIE TAGOWE (BM25-like Scoring)
    # ------------------------------------------------------------------
    criteria_tags = map_phrases_to_tags(tags_queries, model=model_e5, tag_vecs=TAG_VECS, tags_list=TAGS, threshold=0.70)




    query_tag_weights = get_query_tag_weights(criteria_tags)

    n_songs = len(df_songs)
    IDF_MAP = build_idf_map(INV_INDEX, n_songs)

    idf_list = list(IDF_MAP.items())
    sorted_by_idf = sorted(idf_list, key=lambda item: item[1], reverse=True)

    scores_songs_tag = score_songs_by_tags(
        query_tag_weights=query_tag_weights,
        inv_index=INV_INDEX,
        n_songs=len(df_songs),
        idf_map=None,
        use_idf=TAG_SCORING_CONFIG['use_idf'],
        query_pow=TAG_SCORING_CONFIG['query_pow']
    )

    criteria_audio = phrases_to_features(audio_features_queries, lang_code=lang_code)

    # 4. POBIERANIE KANDYDATÓW
    candidates_df = retrieve_candidates_tags(scores_songs_tag)
    print(candidates_df["track_id"].count())

    has_tags = not candidates_df.empty
    if not has_tags and criteria_audio:
        print("Brak wyników z tagów.")
        candidates_df = df_songs.copy()
    elif has_tags:
        print(f"Znaleziono {len(candidates_df)} kandydatów po tagach.")
    else:
        print("Pusto (brak tagów i brak audio).")

    if not candidates_df.empty and criteria_audio:
        candidates_df['audio_score'] = calculate_audio_match(candidates_df, criteria_audio)

    # 5. FAZA 3: OCENA AUDIO I MERGE
    # ------------------------------------------------------------------
    criteria_audio = phrases_to_features(
        audio_features_queries,
        lang_code=lang_code,
        confidence_threshold=0.78
    )

    audio_match_vector = calculate_audio_match(candidates_df, criteria_audio)
    candidates_df['audio_score'] = audio_match_vector

    candidades_merged_score_df = merge_tag_and_audio_scores(candidates_df, audio_weight=0.6, use_tags=has_tags)


    # 6. FAZA 4: TIEROWANIE I MIKSER
    # ------------------------------------------------------------------
    # Uwaga: Używamy teraz dynamicznych progów
    t_high, t_mid = calculate_dynamic_thresholds(
        candidades_merged_score_df,
        high_threshold=SCORE_TIERS_CONFIG['t_high'],
        mid_threshold=SCORE_TIERS_CONFIG['t_mid']
    )

    tier_a, tier_b, tier_c = tier_by_score(candidades_merged_score_df, t_high, t_mid)

    working = build_working_set(
        tier_a, tier_b, tier_c,
        target_pool_size=SCORE_TIERS_CONFIG['min_final'],
        min_required_size=ilosc,
        popularity_rescue_ratio=0.2
    )

    # 7. SAMPLING
    SAMPLING_CONFIG['final_n'] = ilosc

    final_playlist_df = sample_final_songs(
        working,
        POPULARITY_CONFIG,
        SAMPLING_CONFIG
    )

    columns_to_return = [
        'spotify_id', 'name', 'artist',
        'popularity', 'score', 'album_images', 'duration_ms'
    ]

    if final_playlist_df.empty:
        raise HTTPException(status_code=404, detail="Nie udało się wygenerować playlisty. Wynikowy zbiór jest pusty.")

    # Konwertujemy DataFrame do listy słowników, zawierającej tylko wymagane kolumny
    try:
        # Uwaga: Musisz mieć te kolumny w final_playlist_df!
        results = final_playlist_df[columns_to_return].to_dict(orient='records')
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Brak wymaganej kolumny w DataFrame: {e}. Sprawdź pliki Parquet.")

    # FastAPI automatycznie mapuje listę słowników na List[SongResult]
    return results

    # 8. ZWROT WYNIKÓW (ID Spotify)
    '''
    final_ids = final_playlist_df["spotify_id"].tolist()

    if not final_ids:
        raise HTTPException(status_code=404, detail="Nie udało się wygenerować playlisty. Wynikowy zbiór jest pusty.")

    return final_ids
    '''





#Mikolaj
# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# KONFIGURACJA SPOTIFY
# Zakres uprawnień (Scope). Musimy poprosić o prawo do edycji playlist.
SPOTIFY_SCOPE = "playlist-modify-public playlist-modify-private"

def get_spotify_oauth():
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

    # Walidacja konfiguracji
    if not client_id or not client_secret or not redirect_uri:
        raise ValueError(
            "Missing Spotify configuration. Please check your .env file:\n"
            f"SPOTIPY_CLIENT_ID: {'✓' if client_id else '✗'}\n"
            f"SPOTIPY_CLIENT_SECRET: {'✓' if client_secret else '✗'}\n"
            f"SPOTIPY_REDIRECT_URI: {'✓' if redirect_uri else '✗'}"
        )

    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SPOTIFY_SCOPE,
        cache_path=None  # Wyłącz cache - używamy własnego zarządzania tokenami
    )

user_tokens = {}


@app.get("/songs/all", response_model=list[SongSchema])
def read_songs_all(limit: int = 1000, offset: int = 0, db: Session = Depends(get_db)):
    # Dodajemy joinedload do wydajnego pobierania relacji
    songs = db.query(SongModel).options(joinedload(SongModel.tags)).limit(limit).offset(offset).all()
    return songs


@app.get("/")
def root():
    return {"message": "Api  działa"}




# Zmień: @app.get("/songs/{tag_name}", response_model=list[schemas.SongBase])
@app.get("/songs/{tag_name}", response_model=list[SongSchema])
def read_songs_by_tag(
        tag_name: str,
        limit: int = Query(default=10, ge=1),
        db: Session = Depends(get_db)
        ):
    # Używamy JOIN do tabeli TagModel
    query = db.query(SongModel).join(SongModel.tags).filter(
        TagModel.name == tag_name.lower()
    )

    # Optymalizacja
    query = query.options(joinedload(SongModel.tags))

    songs = query.limit(limit).all()
    if not songs:
        detail_msg = f"Nie znaleziono piosenek z tagiem '{tag_name}'"
        raise HTTPException(status_code=404, detail=detail_msg)

    return songs



## //w parametrze ilosc, - tego na razie nie
## wiele argumentów.
## Doker z,
## podmienianie jednego utworu
## filtr po parametrach
## dodanie dockera
## dodanie rzeczy na gita


@app.get("/songs", response_model=list[SongSchema])
def read_songs(
    q: str | None = None,   # <--- ogólny filtr
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(models.Song)

    if q:
        query = query.filter(
            models.Song.name.ilike(f"%{q}%") |
            models.Song.artist.ilike(f"%{q}%") |
            models.Song.tags_list.ilike(f"%{q}%")
        )

    songs = query.limit(limit).offset(offset).all()
    return songs


##-------------------------SPOTIFY CONFIG-------------------------

@app.get("/spotify/config/check")
def check_spotify_config():
    """
    Sprawdza czy konfiguracja Spotify jest poprawna.
    """
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

    return {
        "configured": bool(client_id and client_secret and redirect_uri),
        "client_id": bool(client_id),
        "client_secret": bool(client_secret),
        "redirect_uri": bool(redirect_uri),
        "redirect_uri_value": redirect_uri if redirect_uri else None
    }


##-------------------------SPOTI-------------------------

@app.get("/login")
def login_spotify():
    """
    Krok 1: Przekierowuje użytkownika do logowania w Spotify.
    """
    sp_oauth = get_spotify_oauth()
    # Pobierz URL autoryzacji
    auth_url = sp_oauth.get_authorize_url()
    # Dodaj parametr show_dialog=true aby wymusić wyświetlenie ekranu logowania
    # nawet jeśli użytkownik jest już zalogowany w przeglądarce
    if '?' in auth_url:
        auth_url += '&show_dialog=true'
    else:
        auth_url += '?show_dialog=true'
    return RedirectResponse(auth_url)


@app.get("/callback")
def callback_spotify(code: str):
    """
    Krok 2: Spotify wraca tutaj z kodem. Wymieniamy go na token.
    """
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

    try:
        sp_oauth = get_spotify_oauth()
        token_info = sp_oauth.get_access_token(code, as_dict=True, check_cache=False)

        if not token_info:
            print("ERROR: Failed to get token from Spotify")
            return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=token_failed")

        # Zapisujemy token
        user_tokens['current_user'] = token_info
        print(f"SUCCESS: Token saved for user")

        # Przekieruj z powrotem do frontendu z komunikatem o sukcesie
        return RedirectResponse(f"{frontend_url}/?spotify_auth=success")

    except ValueError as e:
        # Błąd konfiguracji
        print(f"Configuration error: {e}")
        return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=config")

    except Exception as e:
        error_message = str(e)
        print(f"ERROR in callback: {error_message}")

        # Sprawdź czy to błąd invalid_client
        if "invalid_client" in error_message.lower():
            print("ERROR: Invalid client credentials. Check SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in .env")
            return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=invalid_client")

        # Ogólny błąd
        return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=unknown")


@app.get("/auth/status")
def check_auth_status():
    """
    Sprawdza czy użytkownik jest zalogowany do Spotify.
    """
    token_info = user_tokens.get('current_user')
    if not token_info:
        return {"authenticated": False}

    # Sprawdź czy token jest ważny
    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        # Spróbuj odświeżyć token
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            user_tokens['current_user'] = token_info
        except Exception as e:
            print(f"Error refreshing token: {e}")
            user_tokens.pop('current_user', None)
            return {"authenticated": False}

    # Pobierz informacje o użytkowniku
    try:
        sp = spotipy.Spotify(auth=token_info['access_token'])
        user_info = sp.current_user()
        return {
            "authenticated": True,
            "user": {
                "id": user_info.get('id'),
                "display_name": user_info.get('display_name'),
                "email": user_info.get('email')
            }
        }
    except Exception as e:
        error_message = str(e)
        print(f"Error getting user info: {error_message}")

        # Jeśli błąd 403, to prawdopodobnie użytkownik nie jest dodany w Spotify Dashboard
        if "403" in error_message or "not be registered" in error_message:
            user_tokens.pop('current_user', None)
            return {
                "authenticated": False,
                "error": "spotify_user_not_registered",
                "message": "Musisz dodać swojego użytkownika do listy w Spotify Dashboard (Settings → Users and Access)"
            }

        user_tokens.pop('current_user', None)
        return {"authenticated": False}


@app.post("/auth/logout")
def logout_spotify():
    """
    Wylogowuje użytkownika (usuwa token z pamięci).
    """
    user_tokens.pop('current_user', None)
    return {"message": "Wylogowano pomyślnie"}







#Tutaj musi być wsadzane id
MOJA_LISTA_DO_PLAYLISTY = [
    "5cqaG09jwHAyDURuZXViwC",
    "4dDoIid58lgImNuYAxTRyM"
]

PLAYLIST_NAME = "Moja Playlista z Configu"       # Nazwa
PLAYLIST_DESC = "Opis ustawiony w zmiennej globalnej Python" # Opi s
PLAYLIST_PUBLIC = False                          # Czy publiczna? (True/False)

PLAYLIST_COVER_PATH = "cover.jpg"


# Funkcja pomocnicza (musi być w kodzie, żeby zdjęcie działało)
def encode_image_to_base64(image_path: str):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None


@app.post("/create_playlist_hardcoded")
def create_playlist_hardcoded():
    """
    Wersja z zewnętrznymi zmiennymi + okładką + POPRAWNYMI WCIĘCIAMI.
    """

    # A. Autoryzacja
    token_info = user_tokens.get('current_user')
    if not token_info:
        raise HTTPException(status_code=401, detail="Najpierw zaloguj się na /login")

    sp = spotipy.Spotify(auth=token_info['access_token'])
    user_id = sp.current_user()['id']

    # B. Pobranie danych ze zmiennych globalnych
    current_ids = MOJA_LISTA_DO_PLAYLISTY
    pl_name = PLAYLIST_NAME
    pl_public = PLAYLIST_PUBLIC
    pl_desc = PLAYLIST_DESC
    cover_path = PLAYLIST_COVER_PATH

    # C. Pętla przetwarzająca ID
    spotify_uris = []
    for sid in current_ids:
        if "spotify:track:" not in sid:
            spotify_uris.append(f"spotify:track:{sid}")
        else:
            spotify_uris.append(sid)

    # ---------------------------------------------------------
    # KLUCZOWY MOMENT: Tu kończy się pętla.
    # Kod poniżej jest przesunięty w lewo (nie ma wcięcia).
    # Wykona się TYLKO RAZ.
    # ---------------------------------------------------------

    # D. Tworzenie playlisty
    playlist = sp.user_playlist_create(
        user=user_id,
        name=pl_name,
        public=pl_public,
        description=pl_desc
    )

    # E. Dodawanie zdjęcia (jeśli jest w configu)
    cover_msg = "Brak zdjęcia"
    if cover_path:
        img_base64 = encode_image_to_base64(cover_path)
        if img_base64:
            try:
                sp.playlist_upload_cover_image(playlist['id'], img_base64)
                cover_msg = "Zdjęcie dodane"
            except Exception as e:
                cover_msg = f"Błąd zdjęcia: {e}"

    # F. Wrzucanie utworów
    if spotify_uris:
        sp.playlist_add_items(playlist_id=playlist['id'], items=spotify_uris)

    return {
        "status": "Gotowe!",
        "playlist_url": playlist['external_urls']['spotify'],
        "cover_status": cover_msg,
        "tracks_count": len(spotify_uris)
    }