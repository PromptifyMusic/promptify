import difflib
import os
import random
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
from sqlalchemy import text, cast, Float,  select, func
from pgvector.sqlalchemy import Vector
from . import engine_config
import time

DetectorFactory.seed = 0

TAGS_LIST = []
TAG_VECS = None


def initialize_global_tags(db: Session, retries=6, delay=5):
    """
    Pobiera wektory tagów bezpośrednio z kolumny 'tag_embedding' w PostgreSQL (pgvector)
    i ładuje je do pamięci RAM (NumPy) dla szybkiego wyszukiwania.
    """
    global TAGS_LIST, TAG_VECS

    print("[ENGINE]Pobieranie wektorów tagów z Bazy Danych (pgvector)...")

    tags_data = []

    # Pętla ponawiania (Retry Loop)
    for attempt in range(retries):
        tags_data = db.query(models.Tag.name, models.Tag.tag_embedding) \
            .filter(models.Tag.tag_embedding.isnot(None)) \
            .all()

        if tags_data:
            print(f"[ENGINE] Pobrano dane za {attempt + 1}. próbą.")
            break
        else:
            print(f"[ENGINE] ⚠️ Próba {attempt + 1}/{retries}: Baza wektorów jest pusta/niegotowa. Czekam {delay}s...")
            time.sleep(delay)

    if not tags_data:
        print("[ENGINE] BŁĄD KRYTYCZNY: Nie udało się załadować wektorów po wielu próbach.")
        print("[ENGINE] Upewnij się, że skrypt 'update_tag_vectors.py' został wykonany.")
        TAGS_LIST = []
        TAG_VECS = None
        return

    #2. Konwersja danych z Bazy do NumPy

    temp_names = []
    temp_vecs = []

    for name, embedding in tags_data:
        if embedding is None:
            continue
        temp_names.append(name)
        temp_vecs.append(np.array(embedding, dtype=np.float32))

    TAGS_LIST = temp_names
    TAG_VECS = np.array(temp_vecs)

    print(f"[ENGINE]SUKCES: Załadowano {len(TAG_VECS)} wektorów z bazy do RAM-u.")



# device = (
#     "cuda" if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available()
#     else "cpu"
# )
#
# print(f"[ENGINE] Wykryte urządzenie obliczeniowe: {device}")
#


print("[ENGINE] Ładowanie modeli AI")
model_e5 = SentenceTransformer('intfloat/multilingual-e5-base')
model_gliner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")


nlp_pl = spacy.load("pl_core_news_lg")
nlp_en = spacy.load("en_core_web_md")










def create_matcher_for_nlp(nlp_instance):
    matcher = Matcher(nlp_instance.vocab)

    noun_filter = {
        "POS": {"IN": ["NOUN", "PROPN"]},
        "IS_STOP": False,
        "LEMMA": {"NOT_IN": engine_config.GENERIC_LEMMAS}
    }

    matcher.add("FRAZA", [
        [noun_filter],

        [
            {"POS": "ADJ", "LEMMA": {"IN": engine_config.GENRE_MODIFIERS}}, 
            noun_filter
        ],

        [{"POS": "ADV", "OP": "?"}, {"POS": "ADP"}, noun_filter],

        [{"POS": "ADV"}, {"POS": "ADJ", "IS_STOP": False}],

        [{"POS": "ADJ", "IS_STOP": False}],

        [noun_filter, noun_filter],

        [{"POS": "ADV", "OP": "?"}, {"POS": "ADJ"}, {"POS": "ADP"}, noun_filter],

        [
            {"POS": "VERB", "LEMMA": {"NOT_IN": engine_config.GENERIC_VERBS}},
            {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}, "IS_STOP": False, "LEMMA": {"NOT_IN": engine_config.GENERIC_LEMMAS}, "OP": "+"}
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
        if token.text.lower() in engine_config.NEGATION_TERMS:
            return True
    return False




#zmieniam
def extract_relevant_phrases(prompt, current_nlp, current_matcher):
    prompt = prompt.lower()
    prompt_clean = prompt.strip()

    if not prompt_clean:
        return []
    #
    # try:
    #     lang_code = detect(prompt)
    # except:
    #     lang_code = 'pl'
    #
    # if lang_code == 'en':
    #     current_nlp = nlp_en
    #     current_matcher = matcher_en
    #     lang_msg = "EN"
    # else:
    #     current_nlp = nlp_pl
    #     current_matcher = matcher_pl
    #     lang_msg = "PL"

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

    #print(f"[{lang_msg}] Prompt: '{prompt}' \n-> {unique_phrases}")

    return unique_phrases



GLINER_LABELS = [f"{k} ({v['desc']})" for k, v in engine_config.LABELS_CONFIG.items()]
ROUTING_MAP = {k: v['route'] for k, v in engine_config.LABELS_CONFIG.items()}



def get_label_config_lists(config):

    gliner_labels = []
    label_mapping = {}
    for key, value in config.items():
        full_label = f"{key} ({value['desc']})"
        gliner_labels.append(full_label)
        label_mapping[full_label] = key

    return gliner_labels, label_mapping

GLINER_LABELS_LIST, GLINER_LABEL_MAP = get_label_config_lists(engine_config.LABELS_CONFIG)

#nowe

for tag, keywords in engine_config.LANGUAGE_CONFIG.items():
    for word in keywords:
        engine_config.LEMMA_TO_TAG_MAP[word] = tag


for genre, keywords in engine_config.GENRE_LEMMA_CONFIG.items():
    for word in keywords:
        engine_config.LEMMA_TO_GENRE_MAP[word] = genre



def _match_geographical_location(doc_temp, fuzzy_cutoff):
    """
    Sprawdza dopasowanie do lokalizacji geograficznej (język/kraj).

    Returns:
        dict lub None: Słownik z category i route jeśli znaleziono, None w przeciwnym razie.
    """
    for token in doc_temp:
        lemma = token.lemma_.lower()

        found_key = None
        if lemma in engine_config.LEMMA_TO_TAG_MAP:
            found_key = lemma
        elif len(lemma) > 4:
            matches = difflib.get_close_matches(lemma, engine_config.LEMMA_TO_TAG_MAP.keys(), n=1, cutoff=fuzzy_cutoff)
            if matches:
                found_key = matches[0]
                print(f"Naprawiono literówkę: '{lemma}' -> '{found_key}'")

        if found_key:
            return {
                "category": "geographical_location",
                "route": "TAGS"
            }

    return None


def _match_music_genre_exact(phrase_lower):
    """
    Sprawdza dokładne dopasowanie do popularnych gatunków muzycznych.

    Returns:
        dict lub None: Słownik z category i route jeśli znaleziono, None w przeciwnym razie.
    """
    for genre_key, phrases in engine_config.GENRE_PHRASES_EXACT.items():
        for exact_phrase in phrases:
            if exact_phrase in phrase_lower:
                return {
                    "category": "music_genre",
                    "route": "TAGS"
                }
    return None


def _match_music_genre_lemma(doc_temp, fuzzy_cutoff):
    """
    Sprawdza dopasowanie gatunku muzycznego przez lematyzację.

    Returns:
        dict lub None: Słownik z category i route jeśli znaleziono, None w przeciwnym razie.
    """
    for token in doc_temp:
        lemma = token.lemma_.lower()

        # dokładne dopasowanie
        if lemma in engine_config.LEMMA_TO_GENRE_MAP:
            return {
                "category": "music_genre",
                "route": "TAGS"
            }

        # fuzzy match
        elif len(lemma) > 4:
            matches = difflib.get_close_matches(lemma, engine_config.LEMMA_TO_GENRE_MAP.keys(), n=1, cutoff=fuzzy_cutoff)

            if matches:
                found_key = matches[0]
                print(f"Naprawiono literówkę gatunku: '{lemma}' -> '{found_key}'")
                return {
                    "category": "music_genre",
                    "route": "TAGS"
                }

    return None


def _match_time_period(phrase_lower, doc_temp):
    """
    Sprawdza dopasowanie do okresu czasu/dekady.

    Returns:
        dict lub None: Słownik z category i route jeśli znaleziono, None w przeciwnym razie.
    """
    # dekady
    decade_keywords = ["lat", "rok", "80", "90", "00", "70", "60", "19", "20"]
    has_digit = any(char.isdigit() for char in phrase_lower)

    if has_digit and any(x in phrase_lower for x in decade_keywords):
        return {
            "category": "okres_czasu",
            "route": "TAGS"
        }

    # 2ERY / SŁOWA CZASOWE
    era_lemmas = [
        "stary", "nowy", "old", "oldies", "klasyk", "classic",
        "retro", "vintage", "new", "newschool", "oldschool", "współczesny"
    ]

    # Sprawdzamy tokeny ze spaCy
    for token in doc_temp:
        lemma = token.lemma_.lower()
        text = token.text.lower()

        if lemma in era_lemmas or text in era_lemmas:
            return {
                "category": "okres_czasu",
                "route": "TAGS"
            }

    return None


def _match_with_gliner(phrase_lower, gliner_predictions):
    """
    Dopasowuje frazę do encji wykrytych przez GLiNER.

    Returns:
        dict lub None: Słownik z category i route jeśli znaleziono, None w przeciwnym razie.
    """
    for entity in gliner_predictions:
        entity_lower = entity['text'].lower().strip()

        # Sprawdzenie pokrycia (overlap)
        if phrase_lower in entity_lower or entity_lower in phrase_lower:
            full_label = entity['label']
            short_key = GLINER_LABEL_MAP.get(full_label)

            if short_key:
                return {
                    "category": short_key,
                    "route": engine_config.LABELS_CONFIG[short_key]['route']
                }

    return None


def _classify_single_phrase(phrase, doc_temp, gliner_predictions, fuzzy_cutoff):
    """
    Klasyfikuje pojedynczą frazę przez serię dopasowań.

    Returns:
        dict: Słownik z phrase, category i route.
    """
    phrase_lower = phrase.lower().strip()

    # POPULARNE GATUNKI - dokładne dopasowanie
    match = _match_music_genre_exact(phrase_lower)
    if match:
        return {
            "phrase": phrase,
            "category": match["category"],
            "route": match["route"]
        }

    # POPULARNE GATUNKI - lematyzacja
    match = _match_music_genre_lemma(doc_temp, fuzzy_cutoff)
    if match:
        return {
            "phrase": phrase,
            "category": match["category"],
            "route": match["route"]
        }

    # CZAS/DEKADA
    match = _match_time_period(phrase_lower, doc_temp)
    if match:
        return {
            "phrase": phrase,
            "category": match["category"],
            "route": match["route"]
        }

    # JĘZYK/KRAJ
    match = _match_geographical_location(doc_temp, fuzzy_cutoff)
    if match:
        return {
            "phrase": phrase,
            "category": match["category"],
            "route": match["route"]
        }

    # GLINER
    match = _match_with_gliner(phrase_lower, gliner_predictions)
    if match:
        return {
            "phrase": phrase,
            "category": match["category"],
            "route": match["route"]
        }

    # FALLBACK
    return {
        "phrase": phrase,
        "category": "none",
        "route": "AUDIO"
    }


def classify_phrases_with_gliner(prompt, spacy_phrases, model, nlp_model, threshold=0.3, fuzzy_cutoff=0.9):
    """
    Klasyfikuje frazy wykryte przez spaCy, dopasowując je do encji wykrytych przez model GLiNER
    w kontekście całego promptu.

    Funkcja łączy precyzyjne wycinanie fraz (spaCy) z rozumieniem kontekstu (GLiNER).
    Opiera się na globalnych zmiennych konfiguracyjnych: GLINER_LABELS_LIST, GLINER_LABEL_MAP oraz LABELS_CONFIG.

    Args:
        prompt (str): Pełna treść zapytania użytkownika (niezbędna dla kontekstu GLiNERa).
        spacy_phrases (List[str]): Lista fraz (noun chunks/entities) wyekstrahowanych wcześniej przez spaCy.
        model (Any): Załadowany model GLiNER (np. obiekt klasy GLiNER).
        nlp_model: Załadowany model spaCy do lematyzacji.
        threshold (float, optional): Próg pewności dla predykcji GLiNERa. Domyślnie 0.3.
        fuzzy_cutoff (float, optional): Próg dla fuzzy matching. Domyślnie 0.8.

    Returns:
        List[Dict[str, str]]: Lista słowników, gdzie każdy słownik reprezentuje sklasyfikowaną frazę:
            {
                "phrase": str (oryginalna fraza ze spaCy),
                "category": str (klucz kategorii np. 'gatunek_muzyczny' lub 'cecha_audio'),
                "route": str (typ routingu np. 'TAGS' lub 'AUDIO')
            }
    """
    if not spacy_phrases or nlp_model is None:
        return []

    # Uruchamiamy GLiNER na całym tekście
    gliner_predictions = model.predict_entities(prompt, GLINER_LABELS_LIST, threshold=threshold)

    results = []

    for phrase in spacy_phrases:
        phrase_lower = phrase.lower().strip()
        doc_temp = nlp_model(phrase_lower)

        result = _classify_single_phrase(phrase, doc_temp, gliner_predictions, fuzzy_cutoff)
        results.append(result)

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

SEARCH_INDICES = prepare_search_indices(model_e5, engine_config.FEATURE_DESCRIPTIONS, engine_config.ACTIVITY_GROUPS)


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
                    best_audio_val = engine_config.FEATURE_DESCRIPTIONS[feat][idx][0]

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
        if group in engine_config.ACTIVITY_GROUPS:
            for r_feat, r_val in engine_config.ACTIVITY_GROUPS[group]['rules']:
                merged[r_feat] = {'value': r_val, 'confidence': float(score)}

    for feat, val, score in found_explicit_audio:
        if feat not in merged or score > merged[feat]['confidence']:
            if feat in merged:
                print(f"nadpisywanie '{feat}': (Score: {score:.2f})")
            merged[feat] = {'value': val, 'confidence': float(score)}

    print("\n[AUDIO MATCH] Zmapowane cechy auclassify_phrases_with_glinerdio:", flush=True)
    if not merged:
        print("   -> Brak (używam domyślnych/random)", flush=True)
    else:
        for k, v in merged.items():
            print(f"   -> {k}: {v['value']} (Pewność: {v['confidence']:.2f})", flush=True)


    return sorted([(k, v) for k, v in merged.items()], key=lambda x: x[1]['confidence'], reverse=True)



#  MP-------------------------------------------------------------------------------

def map_phrases_to_tags(
        phrases: list[str],
        model=model_e5,
        threshold_strict: float = 0.82,  # Pewniak (E5 mówi: to jest to samo)
        threshold_lenient: float = 0.75  # Ratunek (E5 niepewne, ale tekst podobny)
) -> dict[str, float]:
    # Zabezpieczenie przed pustą bazą/frazami
    if not phrases or TAG_VECS is None or len(TAG_VECS) == 0:
        if TAG_VECS is None:
            print("[ENGINE] OSTRZEŻENIE: Wektory tagów nie są załadowane!")
        return {}

    print(f"\n[ENGINE] Mapowanie Hybrydowe dla fraz: {phrases}")

    # 1. Obliczamy wektory dla fraz użytkownika
    # Używamy modelu E5 załadowanego globalnie
    q_vecs = model.encode(
        [f"query: {p}" for p in phrases],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # 2. Macierz podobieństwa (Wszystkie Tagi z Bazy vs Wszystkie Frazy)
    sims_matrix = cosine_similarity(TAG_VECS, q_vecs)

    found_tags = {}

    for i, phrase in enumerate(phrases):
        # Pobieramy kolumnę wyników dla danej frazy
        phrase_scores = sims_matrix[:, i]

        # Znajdujemy najlepszy tag
        best_idx = np.argmax(phrase_scores)
        best_score = float(phrase_scores[best_idx])
        best_tag_name = TAGS_LIST[best_idx]  # Bierzemy nazwę z globalnej listy

        # --- LOGIKA HYBRYDOWA ---
        is_match = False
        reason = ""

        # A. Czy wynik jest bardzo wysoki? (Pewniak E5)
        # Np. "heavy metal" -> "Metal" (Score 0.85)
        if best_score >= threshold_strict:
            is_match = True
            reason = "High Score"

        # B. Czy wynik jest średni, ale TEKST SIĘ ZGADZA? (Koło ratunkowe)
        # Np. "rockowa" -> "Rock" (Score 0.78 - normalnie by odpadło, ale tekst ratuje)
        elif best_score >= threshold_lenient:
            p_lower = phrase.lower()
            t_lower = best_tag_name.lower()

            # Warunek 1: Zawieranie się (substring)
            if t_lower in p_lower or p_lower in t_lower:
                is_match = True
                reason = "Text Rescue"

            # Warunek 2: Fuzzy Match (literówki, np. "metalica" -> "metal")
            # ratio > 0.7 oznacza 70% zgodności liter
            elif difflib.SequenceMatcher(None, p_lower, t_lower).ratio() > 0.7:
                is_match = True
                reason = "Fuzzy Rescue"

        # --- ZAPISYWANIE WYNIKU ---
        if is_match:
            print(f"MATCH: '{phrase}' -> '{best_tag_name}' ({best_score:.3f}) [{reason}]")
            # Jeśli tag już jest, bierzemy max score
            found_tags[best_tag_name] = max(found_tags.get(best_tag_name, 0), best_score)
        else:
            # Logujemy odrzucenia (dla debugowania)
            if best_score > 0.6:  # Żeby nie śmiecić logami o zerowym dopasowaniu
                print(f"SKIP:  '{phrase}' -> '{best_tag_name}' ({best_score:.3f}) [Za niski wynik]")

    return found_tags




def search_tags_in_db(phrases, db=None):
    return map_phrases_to_tags(phrases)


def get_query_tag_weights(raw_tags)-> dict[str, float]:

    s = sum(raw_tags.values())
    if s <= 0: return raw_tags
    return {t: v / s for t, v in raw_tags.items()}

#najalpej bez limitu




def fetch_candidates_from_db(
        tag_scores: dict[str, float],
        db: Session,
        audio_constraints: list = None,
        limit: int = 2000
) -> pd.DataFrame:

    if limit is None:
        limit = engine_config.RETRIEVAL_CONFIG["n_candidates"]

    songs_query = db.query(models.Song)

    if tag_scores:
        tags_list = list(tag_scores.keys())
        print(f"[DB FETCH] Pobieram po tagach: {tags_list}")
        songs_query = songs_query.join(models.Song.tags).filter(models.Tag.name.in_(tags_list))


    if audio_constraints:
        print(f"[DB FETCH] Filtr cech audio.")

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
                print(f" -> SQL Filter: {feat_name} BETWEEN {safe_min:.2f} AND {safe_max:.2f}")

    songs = []
    # Random sample
    if tag_scores:
        print(f"[DB FETCH] Strategia: RANKING (Najlepsze dopasowania).")
        # KROK 1: Pobieramy TYLKO ID piosenek, żeby uniknąć błędu "GroupingError"
        # Używamy with_entities, żeby nadpisać SELECT na samo ID
        id_query = songs_query.with_entities(models.Song.song_id)

        ranked_ids_tuples = (
            id_query
            .group_by(models.Song.song_id)
            .order_by(
                func.count(models.Song.song_id).desc(),  # Najważniejsze: liczba trafień
                func.random()  # Losowość przy remisie
            )
            .limit(limit)
            .all()
        )
        # Wyciągamy same ID z krotek
        ranked_ids = [r[0] for r in ranked_ids_tuples]

        if ranked_ids:
            # KROK 2: Pobieramy pełne obiekty dla znalezionych ID
            # Joinedload przyspieszy pobieranie tagów
            unordered_songs = db.query(models.Song) \
                .filter(models.Song.song_id.in_(ranked_ids)) \
                .options(joinedload(models.Song.tags)) \
                .all()

            # KROK 3: Przywracamy kolejność rankingu (bo IN_ niszczy kolejność)
            song_map = {s.song_id: s for s in unordered_songs}
            songs = [song_map[pid] for pid in ranked_ids if pid in song_map]

            print(f"[DEBUG] Pobranno {len(songs)} utworów w 2 krokach.")

            #LOG: TOP 5 UTWORÓW I ICH TAG
            print(f"\n[DEBUG] --- TOP 5 KANDYDATÓW (Ranking) ---")
            query_tags_set = set(tag_scores.keys())

            for i, s in enumerate(songs[:5]):
                # Wyciągamy tagi utworu i sprawdzamy część wspólną z zapytaniem
                song_tags = {t.name for t in s.tags}
                matched = song_tags.intersection(query_tags_set)
                # Reszta tagów (dla kontekstu)
                other = list(song_tags - matched)[:3]

                print(f" {i + 1}. {s.artist} - {s.name}")
                print(f"    -> Pasujące tagi ({len(matched)}): {matched}")
                # print(f"    -> Inne tagi: {other}...") # Opcjonalnie
            print(f"----------------------------------------\n")
            # ------------------------------------------



            # Debug log (opcjonalnie)
            if songs:
                top_s = songs[0]
                matching = {t.name for t in top_s.tags}.intersection(tag_scores.keys())
                print(f"[DEBUG] Top 1: {top_s.artist} - {top_s.name} (Tagi: {len(matching)})")



    else:
        print(f"[DB FETCH] Strategia: SAMPLING (Losowa próbka).")


        songs_query = songs_query.distinct()
        total_estimate = songs_query.count()

        if total_estimate <= limit:
            print(f"[DB FETCH] Mała pula ({total_estimate}), biorę wszystko.")
            songs = songs_query.all()

        else:
            seed = random.randint(0, 1_000_000)
            SAMPLE_BUCKETS = 100

            TARGET_BUCKETS = max(
                1, int(100 * (limit * 1.2) / max(total_estimate, 1))
            ) # procent rekordów do pobrania

            print(f"[DB FETCH] Pobieram hash-based sample: {TARGET_BUCKETS}% z {total_estimate} rekordów")

            songs = (
                songs_query
                .filter(
                    func.mod(
                        func.abs(func.hashtext(func.concat(models.Song.song_id, str(seed)))),
                        SAMPLE_BUCKETS
                    ) < TARGET_BUCKETS
                )
                .limit(limit)
                .all()
            )
            if not songs:
                print("[DB FETCH] Fallback: Random limit.")
                songs = songs_query.order_by(func.random()).limit(limit).all()

    if not songs:
        return pd.DataFrame()


    # Tworzenie dataframe
    data = []
    q_pow = engine_config.SCORING_CONFIG.get("query_pow", 1.0)

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
            "n_loudness": float(s.n_loudness or 0.5),
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
