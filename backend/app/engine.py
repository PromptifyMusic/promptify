import os
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, DetectorFactory
from sqlalchemy import text
from sqlalchemy.orm import Session, joinedload, load_only
from . import models
from sqlalchemy import text, cast, Float




DetectorFactory.seed = 0



EXTRACTION_CONFIG = {
    "gliner_threshold": 0.3,
    "tag_similarity_threshold": 0.65,
    "audio_confidence_threshold": 0.78,
}

SCORING_CONFIG = {
    # Tagi
    "use_idf": True,
    "query_pow": 1.0,

    # Fuzja
    "audio_weight": 0.4,

}


RETRIEVAL_CONFIG = {
    "n_candidates": 400,
    "flat_delta": 0.05,
}


WORKSET_CONFIG = {

    "min_absolute_high": 0.75,
    "min_absolute_mid": 0.50,

    "target_pool_size": 100,
    # This does not seem to work as intended. It seems to me that its value sets max size rather than min size.
    # Value set to 50 as temporary workaround. (50 is max size of the playlist I set on frontend)
    "min_required_size": 50,

    "popularity_rescue_ratio": 0.2,
}

POPULARITY_CONFIG = {
    "p_high": 70,
    "p_mid": 35,

    "mix": {
        "high": 0.40,
        "mid":  0.35,
        "low":  0.25,
    },


    "forced_popular": 2,
    "forced_popular_min": 80,
}

SAMPLING_CONFIG = {
    "final_n": 15,
    "alpha": 2.0,
    "shuffle": True,
}



GENERIC_LEMMAS = [
    "music", "song", "track", "playlist", "list", "recording", "audio", "sound", "style", "vibe", "type", "kind", "number", "piece",
    "muzyka", "piosenka", "utwór", "kawałek", "lista", "nagranie", "dźwięk", "gatunek", "styl", "klimat", "typ", "rodzaj"
]





print("[ENGINE] Ładowanie modeli AI")
model_e5 = SentenceTransformer('intfloat/multilingual-e5-base')
model_gliner = GLiNER.from_pretrained("urchade/gliner_small-v2.1")


nlp_pl = spacy.load("pl_core_news_lg")
nlp_en = spacy.load("en_core_web_md")


TAG_VECS = None
TAGS_LIST = None
def initialize_global_tags(db: Session):

    global TAG_VECS, TAGS_LIST
    print("[ENGINE] Pobieranie wektorów tagów z Bazy do RAMu...")

    tags_db = db.query(models.Tag).filter(models.Tag.tag_embedding.isnot(None)).all()

    if tags_db:
        TAGS_LIST = [t.name for t in tags_db]
        TAG_VECS = np.array([t.tag_embedding for t in tags_db], dtype=np.float32)
        print(f"[ENGINE] Sukces: Załadowano {len(TAGS_LIST)} tagów do pamięci.")
    else:
        print("[ENGINE] Ostrzeżenie: Brak tagów w bazie danych!")
        TAGS_LIST = []
        TAG_VECS = np.array([])
#---------------------------------------------------


GENERIC_VERBS = [

    "szukać", "poszukiwać", "chcieć", "pragnąć", "potrzebować", "woleć", "wymagać",

    "być", "mieć", "znajdować", "znaleźć", "słuchać", "posłuchać", "grać", "zależeć",

    "słuchać", "posłuchać", "usłyszeć", "grać", "zagrać", "puszczać", "puścić", "odtworzyć", "zapodać",

    "prosić", "polecić", "polecać", "rekomendować", "sugerować", "zaproponować", "dawać", "dać",

    "search", "look", "find", "want", "need", "desire", "wish", "require",

    "be", "have", "get",

    "listen", "hear", "play", "replay", "stream",

    "give", "recommend", "suggest", "show", "provide",
]

NEGATION_TERMS = [

    "nie", "bez", "mało", "zero", "ani", "żaden", "brak", "mniej",

    "no", "not", "without", "less", "non", "neither", "nor", "lack", "zero"
]



def create_matcher_for_nlp(nlp_instance):
    matcher = Matcher(nlp_instance.vocab)

    noun_filter = {
        "POS": {"IN": ["NOUN", "PROPN"]},
        "IS_STOP": False,
        "LEMMA": {"NOT_IN": GENERIC_LEMMAS}
    }

    matcher.add("FRAZA", [
        [noun_filter],

        [{"POS": "ADJ"}, noun_filter],

        [{"POS": "ADV", "OP": "?"}, {"POS": "ADP"}, noun_filter],

        [{"POS": "ADV"}, {"POS": "ADJ", "IS_STOP": False}],

        [{"POS": "ADJ", "IS_STOP": False}],

        [{"POS": {"IN": ["NOUN", "PROPN"]}, "IS_STOP": False}, noun_filter],

        [{"POS": "ADV", "OP": "?"}, {"POS": "ADJ"}, {"POS": "ADP"}, noun_filter],

        [
            {"POS": "VERB", "LEMMA": {"NOT_IN": GENERIC_VERBS}},
            {"POS": {"IN": ["NOUN", "ADJ", "PRON"]}, "OP": "+"}
        ],

        [{"POS": "ADP"}, {"POS": "NOUN"}, {"IS_DIGIT": True}],
        [{"POS": "NOUN"}, {"IS_DIGIT": True}],

        [{"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}}, {"ORTH": "-"}, {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}}],
    ])
    return matcher

matcher_pl = create_matcher_for_nlp(nlp_pl)
matcher_en = create_matcher_for_nlp(nlp_en)





def is_span_negated(doc, start_index, window=2):

    lookback = max(0, start_index - window)
    preceding_tokens = doc[lookback:start_index]

    for token in preceding_tokens:
        if token.text.lower() in NEGATION_TERMS:
            return True
    return False



def extract_relevant_phrases(prompt):
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

    matcher_matches = current_matcher(doc)

    matcher_spans = [doc[start:end] for match_id, start, end in matcher_matches]

    combined_spans = filter_spans(matcher_spans)

    final_phrases = []

    for span in combined_spans:
        if is_span_negated(doc, span.start):
            continue

        final_phrases.append(span.text.lower())

    unique_phrases = sorted(list(set([p.strip() for p in final_phrases if len(p.strip()) > 2])))

    print(f"[{lang_msg}] Prompt: '{prompt}' \n-> {unique_phrases}")

    return unique_phrases


LABELS_CONFIG = {
    "gatunek_muzyczny": {
        "desc": "rock, pop, jazz, hip hop, metal, indie, alternative, emo, psychedelic, industrial, grunge, punk, pank, postpankowy, post-punk, folk, electronic, experimental, noise music",
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
        "desc": "80s, 90s, 00s, 2020s, oldies, retro, klasyk, lata 90, lata 80, rok 90, rok 80, rok 70",
        "route": "TAGS"
    },
    "pochodzenie": {
        "desc": "polish, american, british, french, k-pop, latino, spanish, deutsch",
        "route": "TAGS"
    },

    "cecha_audio": {
        "desc": "sad, happy, fast, slow, danceable, party, energetic, calm, relaxing, loud, quiet, acoustic, electronic, melancholic, gloomy, euphoric, club banger",
        "route": "AUDIO"
    }
}

GLINER_LABELS = [f"{k} ({v['desc']})" for k, v in LABELS_CONFIG.items()]
ROUTING_MAP = {k: v['route'] for k, v in LABELS_CONFIG.items()}




def get_label_config_lists(config):

    gliner_labels = []
    label_mapping = {}
    for key, value in config.items():
        full_label = f"{key} ({value['desc']})"
        gliner_labels.append(full_label)
        label_mapping[full_label] = key

    return gliner_labels, label_mapping

GLINER_LABELS_LIST, GLINER_LABEL_MAP = get_label_config_lists(LABELS_CONFIG)


def classify_phrases_with_gliner(prompt, spacy_phrases, model, threshold=0.3):
    if not spacy_phrases:
        return []

    gliner_predictions = model.predict_entities(prompt, GLINER_LABELS_LIST, threshold=threshold)

    results = []

    for phrase in spacy_phrases:
        matched_category = None
        matched_route = "AUDIO"

        phrase_lower = phrase.lower().strip()

        best_score = 0

        if any(x in phrase_lower for x in ["lat", "rok", "80", "90", "00", "70"]) and any(char.isdigit() for char in phrase_lower):
            matched_category = "okres_czasu"
            matched_route = "TAGS"
            print(f"[RULE:TIME] '{phrase}' wymuszono kategorię TAGS")

        for entity in gliner_predictions:
            entity_lower = entity['text'].lower().strip()

            if phrase_lower in entity_lower or entity_lower in phrase_lower:
                full_label = entity['label']
                short_key = GLINER_LABEL_MAP.get(full_label)

                if short_key:
                    matched_category = short_key
                    matched_route = LABELS_CONFIG[short_key]['route']
                    break

        if not matched_category:
            matched_category = "cecha_audio"
            matched_route = "AUDIO"

        results.append({
            "phrase": phrase,
            "category": matched_category,
            "route": matched_route
        })
    return results



def prepare_queries_for_e5_separated(classified_data, original_prompt):

    return {
        "TAGS": [
            item['phrase'] for item in classified_data
            if item['route'] == 'TAGS'
        ],
        "AUDIO": [
            item['phrase'] for item in classified_data
            if item['route'] == 'AUDIO'
        ]
    }



FEATURE_DESCRIPTIONS = {
    'valence': [
        # ((Min, Max), "Opis")
        ((0.0, 0.25), "very low valence, very sad, melancholic, dark, gloomy emotional mood music - bardzo smutna, dołująca, ponura, depresyjna, mroczna"),
        ((0.25, 0.40), "low valence, bittersweet, thoughtful, introspective, moody emotional mood music - smutnawa, nostalgiczna, refleksyjna, nastrojowa, melancholijna"),
        # ((0.45, 0.60), "medium valence, neutral emotional mood, neither clearly happy nor clearly sad music"),
        ((0.70, 0.90), "high valence, positive, pleasant, warm, cheerful, uplifting emotional mood music - wesoła, pozytywna, radosna, przyjemna, ciepła, optymistyczna"),
        ((0.90, 1.0), "very high valence, very happy, joyful, exstatic, euphoric, bright, feel-good emotional mood music - bardzo wesoła, euforyczna, ekstatyczna, pełna radości, szczęśliwa")
    ],

    'danceability': [
        ((0.0, 0.25), "very low danceability, not danceable, abstract or experimental, weak or irregular rhythm music - nie do tańca, nietaneczna, nieregularny rytm, abstrakcyjna, bez rytmu"),
        ((0.25, 0.4), "low danceability, little groove, minimal rhythm, not primarily for dancing music - mało taneczna, słaby rytm, raczej do słuchania niż tańczenia"),
        # ((0.50, 0.70), "medium danceability, some groove, steady rhythm music"),
        ((0.80, 0.90), "high danceability, clear beat, strong groove, good for dancing, club-oriented music - taneczna, do tańca, klubowa, dobry rytm, bujająca"),
        ((0.90, 1.0), "very high danceability, strong groove, infectious rhythm, perfect for dancing, party, club banger music - bardzo taneczna, imprezowa, parkietowa, porywająca do tańca, wixa")
    ],

    'acousticness': [
        ((0.0, 0.05), "very low acousticness, fully electronic, synthetic, digital, computer-generated sound music - w pełni elektroniczna, syntetyczna, cyfrowa, syntezatory, techno brzmienie"),
        ((0.05, 0.25), "low acousticness, mostly electronic with some subtle organic or acoustic elements music - głównie elektroniczna, nowoczesne brzmienie"),
        # ((0.45, 0.65), "medium acousticness, balanced mix of acoustic and electronic instruments, hybrid sound music"),
        ((0.65, 0.85), "high acousticness, mostly acoustic, organic, live instruments such as accoustic guitar or piano music - akustyczna, naturalna, żywe instrumenty, gitara, pianino"),
        ((0.85, 1.0), "very high acousticness, fully acoustic, unplugged, natural, organic instruments only music - w pełni akustyczna, bez prądu, unplugged, naturalne brzmienie")
    ],

    'n_tempo': [
        ((0.0, 0.30), "very slow tempo, very slow pace, dragging rhythm music - bardzo wolne tempo, bardzo wolna, ślimacze tempo, ciągnąca się"),
        ((0.30, 0.45), "slow tempo, downtempo, slow pace, relaxed rhythm music - wolne tempo, wolna, spokojny rytm, powolna"),
        ((0.45, 0.7), "medium tempo, moderate pace, walking pace music - średnie tempo, umiarkowana szybkość, normalne tempo"),
        ((0.70, 0.90), "fast tempo, uptempo, quick pace, energetic rhythm music - szybkie tempo, szybka, żwawa, energiczny rytm"),
        ((0.90, 1.0), "very fast tempo, rapid pace, racing rhythm, frantic speed music - bardzo szybkie tempo, bardzo szybka, pędząca, zawrotna prędkość")
    ],

    'instrumentalness': [
        ((0.0, 0.35), "very low instrumentalness, strong presence of vocals and lyrics, clear singing, vocal-focused track - wokalna, z wokalem, śpiewana, piosenka z tekstem, głos"),
        ((0.35, 0.75), "medium instrumentalness, mix of vocals and instrumental sections, vocals present but not constant - mieszana, trochę śpiewania trochę muzyki"),
        ((0.75, 1.0), "very high instrumentalness, fully instrumental track, no vocals, no singing, music without lyrics - instrumentalna, bez słów, bez wokalu, sama muzyka, melodia")
    ],

    'energy': [
        ((0.0, 0.25), "very low energy, motionless, static, sleep-inducing, minimal activity music - bardzo niska energia, statyczna, usypiająca, bez energii, leniwa"),
        ((0.25, 0.45), "low energy, relaxed, laid-back, mellow, slow-moving atmosphere music - niska energia, zrelaksowana, luźna, spokojna, chillout"),
        ((0.45, 0.70), "medium energy, moderate pace, steady rhythm, balanced activity music - średnia energia, umiarkowana, zrównoważona"),
        ((0.70, 0.90), "high energy, active, driving rhythm, fast-paced, stimulating, busy arrangement music - wysoka energia, energetyczna, żywa, pobudzająca, mocna"),
        ((0.90, 1.0), "very high energy, hyperactive, restless, frantic, adrenaline-pumping, non-stop action music - bardzo wysoka energia, wybuchowa, szalona, adrenalina, ogień, pompa")
    ],

    'n_loudness': [
        ((0.0, 0.25), "very low loudness, barely audible, near silence, whisper-like volume, extremely quiet music - bardzo cicha, ledwo słyszalna, szept, cisza"),
        ((0.25, 0.50), "low loudness, soft volume, background level, reduced amplitude, delicate sound music - cicha, delikatna, w tle, miękkie brzmienie"),
        ((0.50, 0.75), "medium loudness, standard volume, normal mastering level music - normalna głośność, standardowa"),
        ((0.75, 0.90), "high loudness, loud volume, amplified sound, noisy, high amplitude music - głośna, hałaśliwa, mocne brzmienie"),
        ((0.90, 1.0), "very high loudness, maximum volume, deafening, high decibels music - bardzo głośna, ogłuszająca, maksymalna głośność, huk")
    ],

    'speechiness': [
        ((0.0, 0.22), "very low speechiness, purely musical track, no spoken words, fully melodic music - muzyka, melodia, śpiew, mało gadania"),
        ((0.22, 0.66), "low speechiness, mostly music with occasional spoken words or short background phrases - muzyka ze wstawkami mowy, rap, hip-hop"),
        ((0.66, 1.0), "medium to high speechiness, balanced mix of speech and music, frequent spoken segments, rap-like or talky structure - dużo gadania, mowa, wywiad, audiobook, podcast, recytacja"),
    ],
}



ACTIVITY_GROUPS = {
    'deep_focus': {
        'triggers': [
            # EN
            "reading", "reading books", "studying", "learning", "homework",
            "focus", "concentration", "deep work", "coding", "programming",
            "writing", "library", "chess", "brainstorming", "thinking",
            "exam preparation", "working", "office work", "study session",
            # PL
            "czytanie", "książki", "nauka", "uczenie się", "praca domowa",
            "skupienie", "koncentracja", "głęboka praca", "programowanie", "kodowanie",
            "pisanie", "biblioteka", "szachy", "myślenie", "egzamin", "sesja",
            "praca biurowa", "do nauki", "do pracy"
        ],
        'rules': [
            ('instrumentalness', (0.8, 1.0)),
            ('speechiness', (0.0, 0.2)),
            ('energy', (0.1, 0.5)),
            ('n_loudness', (0.0, 0.6))
        ]
    },

    'sleep_relax': {
        'triggers': [
            "sleeping", "falling asleep", "insomnia", "nap", "napping",
            "meditation", "meditating", "yoga", "mindfulness", "zen",
            "spa", "massage", "calm down", "anxiety relief", "stress relief",
            "lying in bed", "winding down", "evening relaxation", "chill out",

            "spanie", "sen", "zasypianie", "bezsenność", "drzemka",
            "medytacja", "joga", "uważność", "spa", "masaź",
            "spokój", "stres", "leżenie w łóżku", "wieczorny relaks",
            "odpoczynek", "wyciszenie", "do spania", "kołysanka"
        ],
        'rules': [
            ('energy', (0.0, 0.25)),
            ('n_tempo', (0.0, 0.35)),
            ('n_loudness', (0.0, 0.35)),
            ('acousticness', (0.5, 1.0)),
            ('valence', (0.4, 0.7))
        ]
    },

    'workout_intense': {
        'triggers': [
            "gym", "weightlifting", "crossfit", "boxing", "kickboxing",
            "hiit", "interval training", "sprint", "running fast", "cardio",
            "beast mode", "motivation", "pump up", "hardcore training",
            "powerlifting", "bodybuilding", "marathon training",

            "siłownia", "ciężary", "boks", "interwały", "sprint",
            "bieganie", "szybki bieg", "kardio", "motywacja", "trening",
            "mocny trening", "kulturystyka", "maraton", "ćwiczenia",
            "na siłkę", "pompa"
        ],
        'rules': [
            ('energy', (0.8, 1.0)),
            ('n_tempo', (0.7, 1.0)),
            ('n_loudness', (0.7, 1.0)),
            ('danceability', (0.5, 0.9))
        ]
    },

    'commute_jogging': {
        'triggers': [
            "jogging", "walking", "walking the dog", "commuting", "driving",
            "road trip", "car ride", "night drive", "highway", "bus ride",
            "train ride", "traveling", "subway", "city walk", "bike riding",
            "cycling",

            "jogging", "trucht", "spacer", "spacer z psem", "dojazd", "jazda autem",
            "samochód", "podróż", "nocna jazda", "autostrada", "autobus",
            "pociąg", "metro", "miasto", "rower", "jazda na rowerze",
            "kierowanie", "za kółkiem"
        ],
        'rules': [
            ('energy', (0.5, 0.8)),
            ('n_tempo', (0.45, 0.7)),
            ('valence', (0.4, 0.9))
        ]
    },

    'party_club': {
        'triggers': [

            "party", "house party", "clubbing", "dancing", "dancefloor",
            "friday night", "saturday night", "birthday", "celebration",
            "drinking", "pre-game", "getting ready", "festival", "rave",
            "disco", "summer party", "pool party",

            "impreza", "domówka", "klub", "taniec", "parkiet",
            "piątek wieczór", "sobota", "urodziny", "świętowanie",
            "picie", "bifor", "festiwal", "dyskoteka", "letnia impreza",
            "basen", "do tańca", "wixa"
        ],
        'rules': [
            ('danceability', (0.8, 1.0)),
            ('energy', (0.8, 1.0)),
            ('valence', (0.6, 1.0)),
            ('n_loudness', (0.7, 1.0))
        ]
    },

    'chores_background': {
        'triggers': [

            "cleaning", "cleaning the house", "cooking", "kitchen",
            "doing dishes", "gardening", "chores", "housework",
            "morning coffee", "breakfast", "sunday morning",
            "hanging out", "friends coming over", "dinner party", "barbecue",

            "sprzątanie", "porządki", "gotowanie", "kuchnia",
            "zmywanie", "ogród", "prace domowe", "obowiązki",
            "poranna kawa", "śniadanie", "niedziela rano",
            "spotkanie ze znajomymi", "obiad", "grill", "tło",
            "w tle", "do kawy"
        ],
        'rules': [
            ('energy', (0.4, 0.7)),
            ('valence', (0.5, 0.9)),
            ('danceability', (0.4, 0.8)),
            ('acousticness', (0.2, 0.8))
        ]
    },

    'sad_emotional': {
        'triggers': [

            "sad", "crying", "depression", "depressed", "lonely",
            "heartbreak", "breakup", "missing someone", "rainy day",
            "melancholy", "grieving", "emotional", "moody", "nostalgia",
            "bad day",

            "smutek", "płacz", "depresja", "samotność",
            "złamane serce", "rozstanie", "tęsknota", "deszczowy dzień",
            "melancholia", "żałoba", "emocje", "nostalgia",
            "zły dzień", "doła", "smutna"
        ],
        'rules': [
            ('valence', (0.0, 0.3)),
            ('energy', (0.0, 0.4)),
            ('danceability', (0.0, 0.4)),
            ('n_tempo', (0.0, 0.4))
        ]
    },

    'romance': {
        'triggers': [

            "date night", "romantic dinner",
            "candlelight", "intimacy", "cuddling", "boyfriend", "girlfriend",
            "valentine", "sexy", "seduction", "late night", "bedroom",

            "randka", "romantyczna kolacja",
            "świece", "nastrojowa", "intymność", "przytulanie", "chłopak", "dziewczyna",
            "walentynki", "seks", "sypialnia", "wieczór we dwoje", "miłość"
        ],
        'rules': [
            ('n_tempo', (0.1, 0.5)),
            ('danceability', (0.5, 0.8)),
            ('energy', (0.2, 0.6)),
            ('n_loudness', (0.2, 0.6))
        ]
    },

    'gaming': {
        'triggers': [

            "gaming", "playing games", "esports", "streaming", "twitch",
            "league of legends", "fortnite", "fps", "rpg", "cyberpunk",
            "hacker", "futuristic",

            "granie", "gry", "esport", "stream",
            "strzelanki", "haker", "futurystyczna", "do grania", "gierki"
        ],
        'rules': [
            ('energy', (0.7, 1.0)),
            ('acousticness', (0.0, 0.2)),
            ('speechiness', (0.0, 0.3))
        ]
    }
}


def prepare_search_indices(model, feature_descriptions, activity_groups):

    print("[INIT] Generowanie indeksów wektorowych...")
    indices = {
        'AUDIO': {},
        'ACTIVITY': {}
    }

    for key, descs in feature_descriptions.items():
        passages = [f"passage: {d[1]}" for d in descs]
        indices['AUDIO'][key] = model.encode(passages, normalize_embeddings=True)

    for key, data in activity_groups.items():
        triggers_str = ", ".join(data['triggers'])
        desc_text = f"music for {triggers_str}"

        indices['ACTIVITY'][key] = model.encode([f"passage: {desc_text}"], normalize_embeddings=True)

    print(f"[SUCCESS] Zainicjalizowano {len(indices['AUDIO'])} cech audio i {len(indices['ACTIVITY'])} grup aktywności.")
    return indices

SEARCH_INDICES = prepare_search_indices(model_e5, FEATURE_DESCRIPTIONS, ACTIVITY_GROUPS)


def phrases_to_features(phrases_list, search_indices, lang_code='pl'):

    if not phrases_list: return []

    AUDIO_INDEX = search_indices['AUDIO']
    ACTIVITY_INDEX = search_indices['ACTIVITY']

    try:
        nlp_model = nlp_pl if lang_code == 'pl' else spacy.load("en_core_web_sm")
    except:
        nlp_model = nlp_pl

    found_explicit_audio = []
    found_activities = []

    print(f"\nFRAZY: {phrases_list}")

    for phrase in phrases_list:
        suffix = " muzyka" if lang_code == 'pl' else " music"
        context_vec = model_e5.encode([f"query: {suffix} {phrase}"], normalize_embeddings=True)

        doc = nlp_model(phrase.lower())
        first_token = doc[0]

        is_activity_indicator = first_token.pos_ in ["ADP", "PART", "VERB"]
        is_adj_or_adv = any(t.pos_ in ["ADJ", "ADV"] for t in doc)
        has_noun = any(t.pos_ in ["NOUN", "PROPN"] for t in doc)

        if is_activity_indicator:
            search_scope = "ACTIVITY_ONLY"
        elif is_adj_or_adv and not has_noun:
            search_scope = "AUDIO_ONLY"
        else:
            search_scope = "BOTH"

        best_audio_key, best_audio_val, best_audio_score = None, None, 0.0
        best_act_key, best_act_score = None, 0.0

        if search_scope in ["AUDIO_ONLY", "BOTH"]:
            for feat, embs in AUDIO_INDEX.items():
                sims = cosine_similarity(context_vec, embs)[0]
                idx = np.argmax(sims)
                if sims[idx] > best_audio_score:
                    best_audio_score = sims[idx]
                    best_audio_key = feat
                    best_audio_val = FEATURE_DESCRIPTIONS[feat][idx][0]

        if search_scope in ["ACTIVITY_ONLY", "BOTH"]:
            for group, embs in ACTIVITY_INDEX.items():
                sims = cosine_similarity(context_vec, embs)[0]
                mx = np.max(sims)
                if mx > best_act_score:
                    best_act_score = mx
                    best_act_key = group

        winner_type = None
        final_score = 0.0

        if search_scope == "AUDIO_ONLY":
            winner_type = 'AUDIO'; final_score = best_audio_score
        elif search_scope == "ACTIVITY_ONLY":
            winner_type = 'ACTIVITY'; final_score = best_act_score
        else:
            if best_audio_score > best_act_score:
                winner_type = 'AUDIO'; final_score = best_audio_score
            else:
                winner_type = 'ACTIVITY'; final_score = best_act_score

        if winner_type == 'AUDIO':
            print(f"[MATCH:AUDIO] '{phrase}' -> feature: {best_audio_key} (score: {final_score:.4f})")
            found_explicit_audio.append((best_audio_key, best_audio_val, final_score))
        else:
            print(f"[MATCH:ACTIVITY] '{phrase}' -> category: {best_act_key} (score: {final_score:.4f})")
            found_activities.append((best_act_key, final_score))

    merged = {}

    found_activities.sort(key=lambda x: x[1])
    for group, score in found_activities:
        if group in ACTIVITY_GROUPS:
            for r_feat, r_val in ACTIVITY_GROUPS[group]['rules']:
                merged[r_feat] = {'value': r_val, 'confidence': float(score)}

    for feat, val, score in found_explicit_audio:
        if feat not in merged or score > merged[feat]['confidence']:
            if feat in merged:
                  print(f"nadpisywanie '{feat}': (Score: {score:.2f})")
            merged[feat] = {'value': val, 'confidence': float(score)}

    print("\n[AUDIO MATCH] Zmapowane cechy audio:", flush=True)
    if not merged:
        print("   -> Brak (używam domyślnych/random)", flush=True)
    else:
        for k, v in merged.items():
            print(f"   -> {k}: {v['value']} (Pewność: {v['confidence']:.2f})", flush=True)


    return sorted([(k, v) for k, v in merged.items()], key=lambda x: x[1]['confidence'], reverse=True)



#  MP-------------------------------------------------------------------------------

def map_phrases_to_tags(phrases, threshold=None):
    if threshold is None:
        threshold = EXTRACTION_CONFIG["tag_similarity_threshold"]

    if not phrases or TAG_VECS is None or len(TAG_VECS) == 0:
        return {}

    print(f"[ENGINE] Mapowanie w RAM (NumPy): {phrases}")

    q_vecs = model_e5.encode([f"query: {p}" for p in phrases], convert_to_numpy=True, normalize_embeddings=True)

    sims = cosine_similarity(TAG_VECS, q_vecs)
    found_tags = {}

    for i, phrase in enumerate(phrases):
        col = sims[:, i]
        best_idx = np.argmax(col)
        best_score = float(col[best_idx])
        best_tag = TAGS_LIST[best_idx]

        if best_score >= threshold:
            print(f"   MATCH: '{phrase}' -> '{best_tag}' ({best_score:.3f})")
            found_tags[best_tag] = max(found_tags.get(best_tag, 0), best_score)

    return found_tags



def search_tags_in_db(phrases, db=None):
    return map_phrases_to_tags(phrases)


def get_query_tag_weights(raw_tags):

    s = sum(raw_tags.values())
    if s <= 0: return raw_tags
    return {t: v / s for t, v in raw_tags.items()}


def fetch_candidates_from_db(
        tag_scores: dict[str, float],
        db: Session,
        audio_constraints: list = None,  # NOWY PARAMETR: Wynik z phrases_to_features
        limit: int = None
) -> pd.DataFrame:
    if limit is None:
        limit = RETRIEVAL_CONFIG["n_candidates"]

    songs_query = db.query(models.Song)

    # --- SCENARIUSZ A: MAMY TAGI (Najlepsza opcja) ---
    if tag_scores:
        tags_list = list(tag_scores.keys())
        print(f"[DB FETCH] Scenariusz A: Pobieram po tagach: {tags_list}")

        songs_query = songs_query.join(models.Song.tags) \
            .filter(models.Tag.name.in_(tags_list)) \
            .options(joinedload(models.Song.tags))

    #NIE MA TAGÓW
    elif audio_constraints:
        print(f"[DB FETCH] Brak tagów, filtr cech.")

        MARGIN = 0.15

        for feat_name, data in audio_constraints:

            target_range = data['value']

            if isinstance(target_range, (list, tuple)):
                t_min, t_max = target_range
            else:
                t_min = t_max = float(target_range)

            safe_min = max(0.0, t_min - MARGIN)
            safe_max = min(1.0, t_max + MARGIN)

            if hasattr(models.Song, feat_name):
                column = getattr(models.Song, feat_name)
                songs_query = songs_query.filter(cast(column, Float).between(safe_min, safe_max))
                print(f"   -> SQL Filter: {feat_name} BETWEEN {safe_min:.2f} AND {safe_max:.2f}")

    #Brak filtrów
    else:
        print("[DB FETCH]Random Sample")
        songs_query = songs_query.order_by(text("RANDOM()"))

    songs = songs_query.limit(limit).all()

    #emergancy
    if not songs and audio_constraints and not tag_scores:
        print("[DB FETCH]Filtry zwróciły 0 wyników. Pobieram losowe.")
        songs = db.query(models.Song).order_by(text("RANDOM()")).limit(limit).all()

    if not songs:
        return pd.DataFrame()

    data = []
    q_pow = SCORING_CONFIG.get("query_pow", 1.0)

    for s in songs:
        current_score = 0.0
        song_tag_names = {t.name for t in s.tags}

        for t_name, t_weight in tag_scores.items():
            if t_name in song_tag_names:
                current_score += (t_weight ** q_pow)


        data.append({
            "spotify_id": s.spotify_id,
            "name": s.name,
            "artist": s.artist,
            "popularity": s.popularity or 0,
            "album_images": s.album_images,
            "duration_ms": s.duration_ms,
            "tag_score": current_score,  # Jeśli Scenariusz B, to będzie 0.0, ale to OK
            "energy": float(s.energy) if s.energy is not None else 0.5,
            "danceability": float(s.danceability) if s.danceability is not None else 0.5,
            "valence": float(s.valence) if s.valence is not None else 0.5,
            "acousticness": float(s.acousticness) if s.acousticness is not None else 0.5,
            "instrumentalness": float(s.instrumentalness) if s.instrumentalness is not None else 0.5,
            "tempo": float(s.tempo) if s.tempo is not None else 120.0,
        })

    df = pd.DataFrame(data)
    if not df.empty and df['tag_score'].max() > 0:
        df['tag_score'] /= df['tag_score'].max()

    # Inicjalizacja
    if not df.empty:
        df['score'] = df['tag_score']

    return df
#=========================================================================



def calculate_audio_match(candidates_df, audio_criteria):
    if candidates_df.empty:
        return np.array([])

    if not audio_criteria:
        return np.ones(len(candidates_df))

    total_audio_score = np.ones(len(candidates_df))

    for feature_name, criteria in audio_criteria:
        val_data = criteria['value']


        if isinstance(val_data, (list, tuple)):
            target_min, target_max = val_data
        else:
            target_min = target_max = float(val_data)

        if feature_name not in candidates_df.columns:
            continue

        song_values = candidates_df[feature_name].to_numpy()


        dist_below = np.maximum(0, target_min - song_values)

        dist_above = np.maximum(0, song_values - target_max)

        total_dist = dist_below + dist_above

        sim = np.clip(1.0 - total_dist, 0.0, 1.0)

        total_audio_score *= sim

    return total_audio_score


def merge_tag_and_audio_scores(df, audio_weight=0.3, use_tags=True):
    df = df.copy()
    w = audio_weight if use_tags else 1.0

    df['score'] = (df.get('tag_score', 0) * (1 - w)) + (df.get('audio_score', 0) * w)

    for col in ['tag_score', 'audio_score']:
        if col not in df: df[col] = 0.0

    return df.sort_values('score', ascending=False)



def tier_by_score(candidates: pd.DataFrame, t_high: float, t_mid: float):

    tier_a = candidates[candidates["score"] >= t_high].copy()
    tier_b = candidates[(candidates["score"] < t_high) & (candidates["score"] >= t_mid)].copy()
    tier_c = candidates[candidates["score"] < t_mid].copy()
    return tier_a, tier_b, tier_c




def calculate_dynamic_thresholds(candidates_df, high_threshold=0.75, mid_threshold=0.5):

    if candidates_df.empty:
        return 0.0, 0.0

    max_score = candidates_df['score'].max()

    t_high = max(high_threshold, max_score - 0.1)

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

    working_parts = [tier_a]
    current_count = len(tier_a)

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


    if current_count < min_required_size:
        needed = min_required_size - current_count
        part_c = tier_c.sort_values("score", ascending=False).head(needed)
        working_parts.append(part_c)

    if not working_parts:
        return pd.DataFrame(columns=tier_a.columns)

    working = pd.concat(working_parts, ignore_index=True)

    subset_cols = []

    if 'id' in working.columns:
        subset_cols = ['id']
    elif 'name' in working.columns and 'artist' in working.columns:
        subset_cols = ['name', 'artist']

    if subset_cols:
        working = working.drop_duplicates(subset=subset_cols)
    else:
        pass

    working = working.sort_values("score", ascending=False).reset_index(drop=True)

    return working



def bucket_by_popularity(working: pd.DataFrame, p_high: int, p_mid: int):

    pop_high = working[working["popularity"] >= p_high].copy()
    pop_mid = working[(working["popularity"] < p_high) & (working["popularity"] >= p_mid)].copy()
    pop_low = working[working["popularity"] < p_mid].copy()
    return pop_high, pop_mid, pop_low



def weighted_sample(df: pd.DataFrame, k: int, alpha: float):

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


    if len(working) == 0:
        return working.iloc[0:0].copy()

    final_n = sampling_cfg.get("final_n", 15)
    alpha   = sampling_cfg.get("alpha", 2.0)

    if final_n < 3:
        return weighted_sample(working, final_n, alpha)


    p_high = popularity_cfg.get("p_high", 70)
    p_mid  = popularity_cfg.get("p_mid", 35)
    mix    = popularity_cfg.get("mix", {"high": 0.4, "mid": 0.35, "low": 0.25})

    forced_popular         = popularity_cfg.get("forced_popular", 0)
    forced_popular_min     = popularity_cfg.get("forced_popular_min", p_high)




    pop_high, pop_mid, pop_low = bucket_by_popularity(working, p_high=p_high, p_mid=p_mid)

    final_parts = []

    forced_pool = working[working["popularity"] >= forced_popular_min].copy()

    forced_taken = weighted_sample(forced_pool, forced_popular, alpha)

    final_parts.append(forced_taken)

    used_idx = set(forced_taken.index)
    pop_high = pop_high[~pop_high.index.isin(used_idx)]
    pop_mid  = pop_mid[~pop_mid.index.isin(used_idx)]
    pop_low  = pop_low[~pop_low.index.isin(used_idx)]

    print(f"\n[SAMPLE] Buckety przed losowaniem:", flush=True)
    print(f"   -> High Pop (>={p_high}): {len(pop_high)} utworów", flush=True)
    print(f"   -> Mid Pop  (>={p_mid}):  {len(pop_mid)} utworów", flush=True)
    print(f"   -> Low Pop  (<{p_mid}):   {len(pop_low)} utworów", flush=True)

    n_forced = len(forced_taken)
    remaining = max(0, final_n - n_forced)

    if remaining == 0:
        combined = pd.concat(final_parts, ignore_index=False)
        return combined.sort_values("score", ascending=False).reset_index(drop=True)

    target_high = int(round(remaining * mix.get("high", 0.0)))
    target_mid  = int(round(remaining * mix.get("mid",  0.0)))
    target_low  = remaining - target_high - target_mid

    target_high = min(target_high, len(pop_high))
    target_mid  = min(target_mid, len(pop_mid))
    target_low  = min(target_low, len(pop_low))

    sampled_high = weighted_sample(pop_high, target_high, alpha)
    sampled_mid  = weighted_sample(pop_mid,  target_mid,  alpha)
    sampled_low  = weighted_sample(pop_low,  target_low,  alpha)

    final_parts.extend([sampled_high, sampled_mid, sampled_low])

    combined = pd.concat(final_parts, ignore_index=False)

    if len(combined) < final_n:
        missing = final_n - len(combined)
        used_idx = set(combined.index)

        remaining_pool = working[~working.index.isin(used_idx)]
        extra = weighted_sample(remaining_pool, missing, alpha)

        combined = pd.concat([combined, extra], ignore_index=False)

    do_shuffle = sampling_cfg.get("shuffle", True)
    if do_shuffle:
        combined = combined.sample(frac=1).reset_index(drop=True)
    else:
        combined = combined.sort_values("score", ascending=False).reset_index(drop=True)

    return combined


def tier_by_score(candidates: pd.DataFrame, t_high: float, t_mid: float):
    tier_a = candidates[candidates["score"] >= t_high].copy()
    tier_b = candidates[(candidates["score"] < t_high) & (candidates["score"] >= t_mid)].copy()
    tier_c = candidates[candidates["score"] < t_mid].copy()
    return tier_a, tier_b, tier_c



def calculate_dynamic_thresholds(candidates_df, high_threshold=0.75, mid_threshold=0.5):
    if candidates_df.empty:
        return 0.0, 0.0
    max_score = candidates_df['score'].max()
    t_high = max(high_threshold, max_score - 0.1)
    t_mid = max(mid_threshold, max_score - 0.2)
    return t_high, t_mid
