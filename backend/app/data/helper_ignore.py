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




# 1. EKSTRAKCJA I DOPASOWANIE (NLP / E5 / GLiNER)
EXTRACTION_CONFIG = {
    "gliner_threshold": 0.3,            # Min. pewność klasyfikacji encji GLiNER
    "tag_similarity_threshold": 0.65,   # Min. cosine similarity dla mapowania na tagi (E5)
    "audio_confidence_threshold": 0.78, # Min. pewność mapowania na cechy audio
}

# 2. PUNKTACJA (SCORING)
SCORING_CONFIG = {
    "use_idf": True,    # Ważenie IDF - rzadsze tagi = wyższa waga
    "query_pow": 1.0,   # Wykładnik wag zapytania (>1.0 = większa dyskryminacja)
    "audio_weight": 0.4,# Proporcja audio vs tagi (0.4 = 40/60); auto 1.0 przy braku tagów
}

# 3. POBIERANIE KANDYDATÓW (RETRIEVAL)
RETRIEVAL_CONFIG = {
    "n_candidates": 400,# Liczba utworów w fazie wstępnej selekcji
    "flat_delta": 0.05, # Margines tolerancji dla soft cut-off
}

# 4. TIERY JAKOŚCIOWE I ZBIÓR ROBOCZY
WORKSET_CONFIG = {
    "min_absolute_high": 0.75,      # Dolna granica Tier A (Premium)
    "min_absolute_mid": 0.50,       # Dolna granica Tier B (Standard)
    "target_pool_size": 100,        # Docelowy rozmiar zbioru roboczego
    "min_required_size": 15,        # Minimum przed aktywacją logiki ratunkowej
    "popularity_rescue_ratio": 0.2, # Rezerwacja Tier B dla popularnych utworów
}

# 5. SEGMENTACJA POPULARNOŚCI
POPULARITY_CONFIG = {
    "p_high": 70,           # Próg segmentu High (mainstreamowe)
    "p_mid": 35,            # Próg segmentu Mid (umiarkowana rozpoznawalność)
    "mix": {                # Docelowa dystrybucja w playliście (suma = 1.0)
        "high": 0.40,
        "mid":  0.35,
        "low":  0.25,
    },
    "forced_popular": 2,    # Liczba gwarantowanych kotwic (top hity)
    "forced_popular_min": 80,# Min. popularność dla kotwic
}

# 6. FINALNY SAMPLING
SAMPLING_CONFIG = {
    "final_n": 15,  # Długość playlisty
    "alpha": 2.0,   # Temperatura losowania (0=losowe, >1=faworyzuje wysokie score)
    "shuffle": True,# Permutacja kolejności (False = sortowanie wg score)
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
    "być", "mieć", "znajdować", "znaleźć", "słuchać", "posłuchać", "grać", "zależeć",
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
        ],

        [{"POS": "ADP"}, {"POS": "NOUN"}, {"IS_DIGIT": True}], # np. "z lat 90"
        [{"POS": "NOUN"}, {"IS_DIGIT": True}],

        # Wzorzec na gatunki z myślnikiem:
        [{"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}}, {"ORTH": "-"}, {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}}],
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


# GŁÓWNA FUNKCJA EKSTRAKCJI

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

picked_prompt = "rock"

relevant_phrases = extract_relevant_phrases(picked_prompt)
# extract_and_route(TEST_PROMPT_4)




# KONFIGURACJA DO KLASYFIKACJI FRAZ

# GLiNER ma odróżnić Tag (np. rock) od cech audio (np. szybka).

LABELS_CONFIG = {
    # --- TAGI ---
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

    # --- WSZYSTKO INNE (dane audio) ---
    # Wrzucamy wszystko co jest opisem tutaj.
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
    Przetwarza konfigurację etykiet, tworząc listę sformatowanych stringów dla modelu GLiNER
    oraz mapę powrotną do oryginalnych kluczy.

    Args:
        config (Dict[str, Dict[str, str]]): Słownik konfiguracyjny, w którym:
            - Kluczem (str) jest nazwa kategorii (np. 'gatunek_muzyczny').
            - Wartością (dict) jest słownik zawierający co najmniej klucz 'desc'
              z opisem słownym (np. 'rock, pop...').

    Returns:
        Tuple[List[str], Dict[str, str]]: Krotka zawierająca dwa elementy:
            1. gliner_labels (List[str]): Lista sformatowanych etykiet w postaci "Klucz (Opis)".
               Przykład: ["gatunek_muzyczny (rock, pop...)", ...].
            2. label_mapping (Dict[str, str]): Słownik mapujący pełną etykietę z powrotem na klucz.
               Przykład: {"gatunek_muzyczny (rock, pop...)": "gatunek_muzyczny"}.
    """
    gliner_labels = []
    label_mapping = {} # Mapa {"nazwa (opis)": "nazwa"}

    for key, value in config.items():
        full_label = f"{key} ({value['desc']})"
        gliner_labels.append(full_label)
        label_mapping[full_label] = key

    return gliner_labels, label_mapping

GLINER_LABELS_LIST, GLINER_LABEL_MAP = get_label_config_lists(LABELS_CONFIG)


def classify_phrases_with_gliner(prompt, spacy_phrases, model, threshold=0.3):
    """
    Klasyfikuje frazy wykryte przez spaCy, dopasowując je do encji wykrytych przez model GLiNER
    w kontekście całego promptu.

    Funkcja łączy precyzyjne wycinanie fraz (spaCy) z rozumieniem kontekstu (GLiNER).
    Opiera się na globalnych zmiennych konfiguracyjnych: GLINER_LABELS_LIST, GLINER_LABEL_MAP oraz LABELS_CONFIG.

    Args:
        prompt (str): Pełna treść zapytania użytkownika (niezbędna dla kontekstu GLiNERa).
        spacy_phrases (List[str]): Lista fraz (noun chunks/entities) wyekstrahowanych wcześniej przez spaCy.
        model (Any): Załadowany model GLiNER (np. obiekt klasy GLiNER).
        threshold (float, optional): Próg pewności dla predykcji GLiNERa. Domyślnie 0.3.

    Returns:
        List[Dict[str, str]]: Lista słowników, gdzie każdy słownik reprezentuje sklasyfikowaną frazę:
            {
                "phrase": str (oryginalna fraza ze spaCy),
                "category": str (klucz kategorii np. 'gatunek_muzyczny' lub 'cecha_audio'),
                "route": str (typ routingu np. 'TAGS' lub 'AUDIO')
            }
    """
    if not spacy_phrases:
        return []

    # Uruchamiamy GLiNER na całym tekście
    # Dzięki temu odróżni "Rock" (gatunek) od "Szybka" (audio)
    gliner_predictions = model.predict_entities(prompt, GLINER_LABELS_LIST, threshold=threshold)

    results = []

    # Iterujemy po frazach ze spaCy i szukamy dla nich etykiety w wynikach GLiNERa
    for phrase in spacy_phrases:
        matched_category = None
        matched_route = "AUDIO" # Domyślny routing

        # Normalizacja frazy spaCy do porównania
        phrase_lower = phrase.lower().strip()

        # Szukamy czy ta fraza została też znaleziona przez GLiNERa
        # Sprawdzamy czy tekst encji GLiNERa zawiera się w frazie spaCy lub odwrotnie
        best_score = 0

        if any(x in phrase_lower for x in ["lat", "rok", "80", "90", "00", "70"]) and any(char.isdigit() for char in phrase_lower):
            matched_category = "okres_czasu"
            matched_route = "TAGS"
            print(f"[RULE:TIME] '{phrase}' wymuszono kategorię TAGS")

        for entity in gliner_predictions:
            entity_lower = entity['text'].lower().strip()

            # Sprawdzenie pokrycia (overlap)
            if phrase_lower in entity_lower or entity_lower in phrase_lower:
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




classified_phrases = classify_phrases_with_gliner(picked_prompt, relevant_phrases, model_gliner)
classified_phrases




def prepare_queries_for_e5_separated(classified_data, original_prompt):
    """
    Rozdziela frazy na Tagi i Audio.
    Zwraca słownik z dwoma listami: "TAGS" i "AUDIO".
    """

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



e5_queries_separated = prepare_queries_for_e5_separated(classified_phrases, picked_prompt)
e5_queries_separated



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
        # Speechiness > 0.66 to zazwyczaj podcasty, 0.33-0.66 to rap, < 0.33 to muzyka
        ((0.0, 0.22), "very low speechiness, purely musical track, no spoken words, fully melodic music - muzyka, melodia, śpiew, mało gadania"),
        ((0.22, 0.66), "low speechiness, mostly music with occasional spoken words or short background phrases - muzyka ze wstawkami mowy, rap, hip-hop"),
        ((0.66, 1.0), "medium to high speechiness, balanced mix of speech and music, frequent spoken segments, rap-like or talky structure - dużo gadania, mowa, wywiad, audiobook, podcast, recytacja"),
    ],

    # 'noise': [
    #     # Rzeczowniki (generyczne słowa)
    #     (None, "music song track playlist list recording audio sound genre style vibe type kind number piece"),

    #     # Czasowniki (związane z szukaniem)
    #     (None, "I am looking for I want I need search find play listen to give me recommend show me"),

    #     # Przymiotniki (fillery bez konkretnej treści)
    #     (None, "good very good nice great best cool amazing awesome some any kind of such a")
    # ]
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
            # EN
            "sleeping", "falling asleep", "insomnia", "nap", "napping",
            "meditation", "meditating", "yoga", "mindfulness", "zen",
            "spa", "massage", "calm down", "anxiety relief", "stress relief",
            "lying in bed", "winding down", "evening relaxation", "chill out",
            # PL
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
            # EN
            "gym", "weightlifting", "crossfit", "boxing", "kickboxing",
            "hiit", "interval training", "sprint", "running fast", "cardio",
            "beast mode", "motivation", "pump up", "hardcore training",
            "powerlifting", "bodybuilding", "marathon training",
            # PL
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
            # EN
            "jogging", "walking", "walking the dog", "commuting", "driving",
            "road trip", "car ride", "night drive", "highway", "bus ride",
            "train ride", "traveling", "subway", "city walk", "bike riding",
            "cycling",
            # PL
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
            # EN
            "party", "house party", "clubbing", "dancing", "dancefloor",
            "friday night", "saturday night", "birthday", "celebration",
            "drinking", "pre-game", "getting ready", "festival", "rave",
            "disco", "summer party", "pool party",
            # PL
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
            # EN
            "cleaning", "cleaning the house", "cooking", "kitchen",
            "doing dishes", "gardening", "chores", "housework",
            "morning coffee", "breakfast", "sunday morning",
            "hanging out", "friends coming over", "dinner party", "barbecue",
            # PL
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
            # EN
            "sad", "crying", "depression", "depressed", "lonely",
            "heartbreak", "breakup", "missing someone", "rainy day",
            "melancholy", "grieving", "emotional", "moody", "nostalgia",
            "bad day",
            # PL
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
            # EN
            "date night", "romantic dinner",
            "candlelight", "intimacy", "cuddling", "boyfriend", "girlfriend",
            "valentine", "sexy", "seduction", "late night", "bedroom",
            # PL
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
            # EN
            "gaming", "playing games", "esports", "streaming", "twitch",
            "league of legends", "fortnite", "fps", "rpg", "cyberpunk",
            "hacker", "futuristic",
            # PL
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
    """
    Inicjalizuje i generuje wektorowe indeksy wyszukiwania dla cech audio oraz aktywności.

    Proces polega na transformacji opisów tekstowych (natural language descriptions)
    na ustandaryzowane osadzenia wektorowe (embeddings) przy użyciu modelu E5.

    Args:
        model: Model SentenceTransformer (np. multilingual-e5).
        feature_descriptions (dict): Słownik definicji cech technicznych (np. tempo, energy).
        activity_groups (dict): Słownik grup aktywności wraz z ich triggerami.

    Returns:
        dict: Słownik zawierający wygenerowane macierze wektorowe podzielone na AUDIO i ACTIVITY.
    """
    print("[INIT] Generowanie indeksów wektorowych...")
    indices = {
        'AUDIO': {},
        'ACTIVITY': {}
    }

    # 1. Generowanie indeksów AUDIO (Cechy techniczne)
    # Dla każdej cechy generujemy wektory na podstawie opisów poziomów (np. low/high tempo)
    for key, descs in feature_descriptions.items():
        passages = [f"passage: {d[1]}" for d in descs]
        indices['AUDIO'][key] = model.encode(passages, normalize_embeddings=True)

    # 2. Generowanie indeksów ACTIVITY (Scenariusze użytkowania)
    # Tworzymy syntetyczny opis kontekstowy na podstawie triggerów danej aktywności
    for key, data in activity_groups.items():
        triggers_str = ", ".join(data['triggers'])
        desc_text = f"music for {triggers_str}"

        indices['ACTIVITY'][key] = model.encode([f"passage: {desc_text}"], normalize_embeddings=True)

    print(f"[SUCCESS] Zainicjalizowano {len(indices['AUDIO'])} cech audio i {len(indices['ACTIVITY'])} grup aktywności.")
    return indices

# --- SETUP ---
# Uruchamiane raz przy starcie systemu
SEARCH_INDICES = prepare_search_indices(model_e5, FEATURE_DESCRIPTIONS, ACTIVITY_GROUPS)




def phrases_to_features(phrases_list, search_indices, lang_code='pl'):
    """
    Mapuje listę fraz tekstowych na parametry techniczne (audio features) oraz aktywności,
    wykorzystując hybrydowe podejście oparte na analizie gramatycznej (NLP)
    oraz osadzeniach wektorowych (embeddings) modelu E5.

    Logika działania:
    1. Analiza gramatyczna (Routing): Każda fraza jest analizowana przez model spaCy.
       Na podstawie części mowy (POS) przypisywany jest zakres wyszukiwania:
       - ACTIVITY_ONLY: Dla przyimków i czasowników (np. "do spania", "biegać").
       - AUDIO_ONLY: Dla czystych przymiotników i przysłówków (np. "szybka").
       - BOTH: Dla rzeczowników i fraz mieszanych (np. "tempo", "sen").
    2. Wyszukiwanie semantyczne: Obliczane jest podobieństwo kosinusowe między wektorem
       zapytania a wektorami w wybranych indeksach (AUDIO/ACTIVITY).
    3. Turniej (Max Pooling): W trybie BOTH wygrywa kategoria o najwyższym współczynniku ufności.
    4. Konsolidacja (Merge): Parametry audio nadpisują domyślne ustawienia aktywności,
       jeśli ich pewność (confidence) jest wyższa.

    Args:
        phrases_list (list[str]): Lista fraz wyodrębnionych z promptu (np. ["szybka", "do nauki"]).
        search_indices (dict): Słownik zawierający wstępnie obliczone embeddingi dla
            kluczy 'AUDIO' i 'ACTIVITY'.
        lang_code (str, optional): Kod języka ('pl' lub 'en') decydujący o modelu spaCy
            i sufiksie kontekstowym. Domyślnie 'pl'.

    Returns:
        list[tuple]: Posortowana malejąco lista krotek (nazwa_cechy, dane), gdzie dane
            zawierają docelowy przedział wartości i współczynnik ufności.
            Przykład: [('n_tempo', {'value': (0.8, 0.9), 'confidence': 0.88})]
    """

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

        # 1. ANALIZA GRAMATYCZNA (NLP)
        doc = nlp_model(phrase.lower())
        first_token = doc[0]

        # Flagi gramatyczne
        is_activity_indicator = first_token.pos_ in ["ADP", "PART", "VERB"]
        is_adj_or_adv = any(t.pos_ in ["ADJ", "ADV"] for t in doc)
        has_noun = any(t.pos_ in ["NOUN", "PROPN"] for t in doc)

        # 2. ROUTING
        if is_activity_indicator:
            # "do spania", "for sleep", "biegać"
            search_scope = "ACTIVITY_ONLY"
        elif is_adj_or_adv and not has_noun:
            # "szybka", "głośno", "wolno" (bez rzeczownika typu 'tempo')
            search_scope = "AUDIO_ONLY"
        else:
            # "tempo", "sen", "bas", "malowanie", "wysoka energia"
            search_scope = "BOTH"

        # 3. WYSZUKIWANIE
        best_audio_key, best_audio_val, best_audio_score = None, None, 0.0
        best_act_key, best_act_score = None, 0.0

        # Szukamy w Audio
        if search_scope in ["AUDIO_ONLY", "BOTH"]:
            for feat, embs in AUDIO_INDEX.items():
                sims = cosine_similarity(context_vec, embs)[0]
                idx = np.argmax(sims)
                if sims[idx] > best_audio_score:
                    best_audio_score = sims[idx]
                    best_audio_key = feat
                    best_audio_val = FEATURE_DESCRIPTIONS[feat][idx][0]

        # Szukamy w Activity
        if search_scope in ["ACTIVITY_ONLY", "BOTH"]:
            for group, embs in ACTIVITY_INDEX.items():
                sims = cosine_similarity(context_vec, embs)[0]
                mx = np.max(sims)
                if mx > best_act_score:
                    best_act_score = mx
                    best_act_key = group

        # 4. WERDYKT (Max Pooling / Turniej)
        winner_type = None
        final_score = 0.0

        if search_scope == "AUDIO_ONLY":
            winner_type = 'AUDIO'; final_score = best_audio_score
        elif search_scope == "ACTIVITY_ONLY":
            winner_type = 'ACTIVITY'; final_score = best_act_score
        else:
            # TURNIEJ DLA "BOTH": Wygrywa wyższy confidence
            if best_audio_score > best_act_score:
                winner_type = 'AUDIO'; final_score = best_audio_score
            else:
                winner_type = 'ACTIVITY'; final_score = best_act_score

        # Dodawanie do tymczasowych list wyników
        if winner_type == 'AUDIO':
            print(f"[MATCH:AUDIO] '{phrase}' -> feature: {best_audio_key} (score: {final_score:.4f})")
            found_explicit_audio.append((best_audio_key, best_audio_val, final_score))
        else:
            print(f"[MATCH:ACTIVITY] '{phrase}' -> category: {best_act_key} (score: {final_score:.4f})")
            found_activities.append((best_act_key, final_score))

    # 5. ŁĄCZENIE (MERGE) - Audio ma priorytet przy konfliktach
    merged = {}

    # KROK A: Presety aktywności (baza)
    found_activities.sort(key=lambda x: x[1]) # od najsłabszej do najsilniejszej
    for group, score in found_activities:
        if group in ACTIVITY_GROUPS:
            for r_feat, r_val in ACTIVITY_GROUPS[group]['rules']:
                merged[r_feat] = {'value': r_val, 'confidence': float(score)}

    # KROK B: Cechy audio (nadpisują bazę, jeśli są silniejsze)
    for feat, val, score in found_explicit_audio:
        if feat not in merged or score > merged[feat]['confidence']:
            if feat in merged:
                  print(f"nadpisywanie '{feat}': (Score: {score:.2f})")
            merged[feat] = {'value': val, 'confidence': float(score)}

    return sorted([(k, v) for k, v in merged.items()], key=lambda x: x[1]['confidence'], reverse=True)



# extracted_phrases = extract_relevant_phrases(picked_prompt)
audio_features_queries = e5_queries_separated['AUDIO']

# Testowe przykłady
# picked_prompt = 'spanie'
# extracted_phrases = ['szybka', 'do upierdalania opla', 'do grania', 'liga legend', 'do grania w lola', 'jsdgfbdc sdo', 'szybkie tempo', 'z szybkim tempem', 'gówno', 'dupa', 'dziwka', 'wolna', 'głośna', 'wolę', 'lubię', 'zajebista', 'muzyka', 'do spania', 'sleep', 'sleeping', 'do nauki', 'praca', 'do pracy', 'malowanie']

# Wykrywamy język z promptu dla potrzeb suffixu "muzyka/music"
try:
    lang = detect(picked_prompt)
except:
    lang = 'pl'

criteria_audio = phrases_to_features(audio_features_queries, SEARCH_INDICES, lang_code=lang)
criteria_audio



df_tag_embeddings = pd.read_parquet("../df_unique_tag_embeddings.parquet")
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
            # Jeśli inny tag już był zapisany (np. inna fraza też wskazała na "rock"),
            # bierzemy ten z wyższym scorem (Max Pooling)
            if best_tag_name in found_tags:
                found_tags[best_tag_name] = max(found_tags[best_tag_name], best_score)
            else:
                found_tags[best_tag_name] = best_score
        else:
            print(f"    Fraza '{phrase}' nie pasuje do żadnego tagu (max: {best_score:.3f})")
            pass

    return found_tags



tags_queries = e5_queries_separated['TAGS']
print(tags_queries)
criteria_tags = map_phrases_to_tags(tags_queries, model=model_e5, tag_vecs=TAG_VECS, tags_list=TAGS, threshold=0.70)
criteria_tags



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




query_tag_weights = get_query_tag_weights(criteria_tags)
query_tag_weights




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

        # Dodajemy do mianownika: idealna piosenka miałaby wszystkie te tagi
        max_theoretical_score += final_weight

    # Inicjalizacja wektora wyników
    scores = np.zeros(n_songs, dtype=np.float32)

    # Sumowanie punktów (coverage)
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



n_songs = len(df_songs)
IDF_MAP = build_idf_map(INV_INDEX, n_songs)

idf_list = list(IDF_MAP.items())
sorted_by_idf = sorted(idf_list, key=lambda item: item[1], reverse=True)
# sorted_by_idf



scores_songs_tag = score_songs_by_tags(
    query_tag_weights=query_tag_weights,
    inv_index=INV_INDEX,
    n_songs=len(df_songs),
    idf_map=None,
    use_idf=SCORING_CONFIG['use_idf'],
    query_pow=SCORING_CONFIG['query_pow']
)
scores_songs_tag



def retrieve_candidates_tags(
    scores_songs_tag: np.ndarray,
    n_candidates: int = 400,
    flat_delta: float = 0.05
):
    """
    Wybiera kandydatów z bazy utworów na podstawie obliczonych punktów (score) za tagi.
    Stosuje mechanizm "miękkiego odcięcia" (soft cut-off) – jeśli utwory tuż za progiem n_candidates
    mają bardzo zbliżony wynik, również są dołączane do listy.

    Funkcja polega na globalnej zmiennej:
        - df_songs (pd.DataFrame): Główna baza danych utworów.

    Args:
        scores_songs_tag (np.ndarray): Wektor wyników dopasowania tagów dla wszystkich utworów
            (musi mieć tę samą długość co df_songs).
        n_candidates (int, optional): Docelowa liczba kandydatów do zwrócenia. Domyślnie 400.
        flat_delta (float, optional): Margines tolerancji dla wyników. Utwory poza top N,
            których wynik różni się od ostatniego zaakceptowanego o mniej niż ta wartość,
            zostaną również dołączone. Domyślnie 0.05.

    Returns:
        pd.DataFrame: Podzbiór `df_songs` zawierający najlepszych kandydatów,
        z dodanymi kolumnami 'tag_score' oraz 'score'.
    """
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
    Oblicza stopień dopasowania kandydatów do zadanych kryteriów audio (np. energy, danceability).
    Stosuje podejście iloczynowe: ocena końcowa to iloczyn dopasowań dla poszczególnych cech.

    Logika:
    1. Dla każdego kryterium obliczany jest dystans: |wartość_utworu - wartość_docelowa|.
    2. Dystans zamieniany jest na podobieństwo: 1.0 - dystans.
    3. Wyniki cząstkowe są mnożone (utwór musi być bliski wszystkim kryteriom naraz).

    Args:
        candidates_df (pd.DataFrame): DataFrame z utworami. Musi zawierać kolumny odpowiadające
            nazwom cech w audio_criteria (np. 'energy', 'tempo').
        audio_criteria (List[Tuple[str, Dict[str, float]]]): Lista krotek z kryteriami.
            Format: [('energy', {'value': 0.8, ...}), ('danceability', {'value': 0.5})].

    Returns:
        np.ndarray: Wektor o długości równej liczbie wierszy w candidates_df,
        zawierający wartości zmiennoprzecinkowe z zakresu 0.0 - 1.0.
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
        val_data = criteria['value']

        # ZABEZPIECZENIE:
        # Sprawdzamy czy w konfiguracji jest liczba (stara wersja) czy krotka/lista (nowa wersja)
        if isinstance(val_data, (list, tuple)):
            target_min, target_max = val_data
        else:
            # Jeśli user podał jedną liczbę, traktujemy to jako punkt (min=max)
            target_min = target_max = float(val_data)

        # Sprawdzamy, czy cecha istnieje w DataFrame
        if feature_name not in candidates_df.columns:
            continue

        # Pobieramy wartości z kolumny
        song_values = candidates_df[feature_name].to_numpy()

        # OBLICZANIE DYSTANSU (Similarity)
        # 1. Obliczamy o ile wartość poniżej min
        dist_below = np.maximum(0, target_min - song_values)

        # 2. Obliczamy o ile wartość jest powyżej max
        dist_above = np.maximum(0, song_values - target_max)

        # 3. Sumujemy dystans
        total_dist = dist_below + dist_above

        # 4. Zamieniamy na podobieństwo (1.0 - dystans)
        sim = np.clip(1.0 - total_dist, 0.0, 1.0)

        # Mnożymy (intersekcja warunków): Song musi spełniać Warunek 1 i Warunek 2
        total_audio_score *= sim

    return total_audio_score




def calculate_audio_match(candidates_df, audio_criteria):
    """
    Oblicza stopień dopasowania kandydatów do zadanych kryteriów audio (np. energy, danceability).
    Stosuje podejście iloczynowe: ocena końcowa to iloczyn dopasowań dla poszczególnych cech.

    Logika:
    1. Dla każdego kryterium obliczany jest dystans: |wartość_utworu - wartość_docelowa|.
    2. Dystans zamieniany jest na podobieństwo: 1.0 - dystans.
    3. Wyniki cząstkowe są mnożone (utwór musi być bliski wszystkim kryteriom naraz).

    Args:
        candidates_df (pd.DataFrame): DataFrame z utworami. Musi zawierać kolumny odpowiadające
            nazwom cech w audio_criteria (np. 'energy', 'tempo').
        audio_criteria (List[Tuple[str, Dict[str, float]]]): Lista krotek z kryteriami.
            Format: [('energy', {'value': 0.8, ...}), ('danceability', {'value': 0.5})].

    Returns:
        np.ndarray: Wektor o długości równej liczbie wierszy w candidates_df,
        zawierający wartości zmiennoprzecinkowe z zakresu 0.0 - 1.0.
    """
    if candidates_df.empty:
        return np.array([])

    if not audio_criteria:
        return np.ones(len(candidates_df))

    total_audio_score = np.ones(len(candidates_df))

    for feature_name, criteria in audio_criteria:
        val_data = criteria['value']

        # ZABEZPIECZENIE:
        # Sprawdzamy czy w konfiguracji jest liczba (stara wersja) czy krotka/lista (nowa wersja)
        if isinstance(val_data, (list, tuple)):
            target_min, target_max = val_data
        else:
            # Jeśli user podał jedną liczbę, traktujemy to jako punkt (min=max)
            target_min = target_max = float(val_data)

        if feature_name not in candidates_df.columns:
            continue

        song_values = candidates_df[feature_name].to_numpy()

        # === NOWA LOGIKA MATEMATYCZNA ===

        # 1. Obliczamy o ile wartość jest ZA MAŁA (poniżej min)
        # Jeśli piosenka ma 0.5, a min to 0.7 -> diff = 0.2
        # Jeśli piosenka ma 0.75, a min to 0.7 -> diff = -0.05 -> bierzemy 0 (bo maximum(0, ...))
        dist_below = np.maximum(0, target_min - song_values)

        # 2. Obliczamy o ile wartość jest ZA DUŻA (powyżej max)
        # Jeśli piosenka ma 0.9, a max to 0.8 -> diff = 0.1
        dist_above = np.maximum(0, song_values - target_max)

        # 3. Sumujemy dystans (zawsze jedna z tych wartości będzie zerem, albo obie jeśli trafiliśmy w środek)
        total_dist = dist_below + dist_above

        # 4. Zamieniamy na podobieństwo (1.0 - dystans)
        sim = np.clip(1.0 - total_dist, 0.0, 1.0)

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




candidades_df = retrieve_candidates_tags(
    scores_songs_tag,
    n_candidates=RETRIEVAL_CONFIG['n_candidates'],
    flat_delta=RETRIEVAL_CONFIG['flat_delta']
)
candidades_df



has_tags = not candidades_df.empty

# Fallback
if not has_tags and criteria_audio:
    print("Brak wyników z tagów.")
    candidades_df = df_songs.copy()
elif has_tags:
    print(f"Znaleziono {len(candidades_df)} kandydatów po tagach.")
else:
    print("Pusto (brak tagów i brak audio).")

print(picked_prompt)



# Policz Audio Score
if not candidades_df.empty and criteria_audio:
    candidades_df['audio_score'] = calculate_audio_match(candidades_df, criteria_audio)



audio_match_vector = calculate_audio_match(candidades_df, criteria_audio)
candidades_df['audio_score'] = audio_match_vector
candidades_df[['artist', 'name', 'audio_score']].sample(3)


picked_prompt



candidades_merged_score_df = merge_tag_and_audio_scores(candidades_df, audio_weight=0.6, use_tags=has_tags)
candidades_merged_score_df[['artist', 'name', 'tag_score', 'audio_score', 'score']].head(20)



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

    Returns:
        tuple: (tier_a, tier_b, tier_c) — trzy DataFramey z podziałem wg jakości.
    """
    tier_a = candidates[candidates["score"] >= t_high].copy()
    tier_b = candidates[(candidates["score"] < t_high) & (candidates["score"] >= t_mid)].copy()
    tier_c = candidates[candidates["score"] < t_mid].copy()
    return tier_a, tier_b, tier_c



def calculate_dynamic_thresholds(candidates_df, high_threshold=0.75, mid_threshold=0.5):
    """
    Wylicza adaptacyjne progi punktowe (t_high, t_mid) służące do podziału wyników na Tiery (np. A i B).

    Logika łączy podejście względne (relative) z bezwzględnym (absolute):
    1. Podejście względne: Próg dostosowuje się do najlepszego wyniku w danej puli (max_score).
       Jeśli mamy świetne dopasowania (np. 0.95), próg Tier A rośnie (np. do 0.85), aby wybrać tylko "śmietankę".
    2. Podejście bezwzględne (Quality Gate): Próg nigdy nie spadnie poniżej ustalonego minimum (high_threshold).
       Jeśli najlepszy wynik to tylko 0.6, a high_threshold to 0.75, to t_high wyniesie 0.75.
       W efekcie Tier A pozostanie pusty, co jest pożądanym zachowaniem (nie pokazujemy słabych wyników jako "świetne").

    Args:
        candidates_df (pd.DataFrame): DataFrame z kandydatami, musi zawierać kolumnę 'score'.
        high_threshold (float, optional): Minimalna wartość bezwzględna dla Tier A. Domyślnie 0.75.
        mid_threshold (float, optional): Minimalna wartość bezwzględna dla Tier B. Domyślnie 0.5.

    Returns:
        Tuple[float, float]: Krotka zawierająca wyliczone progi (t_high, t_mid).
    """
    if candidates_df.empty:
        return 0.0, 0.0

    # Jaki jest absolutnie najlepszy wynik dla tego zapytania?
    max_score = candidates_df['score'].max()

    # LOGIKA ADAPTACYJNA

    # 1. Tier A: Musi być blisko lidera (np. max - 0.1),
    #    ale nie może być mniejszy niż 0.75.
    #    Jeśli max_score to 0.5, to t_high będzie 0.75 -> Tier A będzie pusty
    t_high = max(high_threshold, max_score - 0.1)

    # 2. Tier B: Musi być sensowny (np. max - 0.3),
    #    ALE nie mniejszy niż 0.4 (żeby nie brać śmieci).
    t_mid = max(mid_threshold, max_score - 0.2)

    return t_high, t_mid




t_high, t_mid = calculate_dynamic_thresholds(
    candidades_merged_score_df,
    high_threshold=WORKSET_CONFIG['min_absolute_high'],
    mid_threshold=WORKSET_CONFIG['min_absolute_mid']
)
t_high, t_mid



tier_a, tier_b, tier_c = tier_by_score(
    candidades_merged_score_df,
    t_high,
    t_mid
)
print(len(tier_a), len(tier_b), len(tier_c))





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

    # 1. zawsze bierzemy całe Tier A
    working_parts = [tier_a]
    current_count = len(tier_a)

    # Jeśli mamy mniej kandydatów niż wynosi cel (target_pool_size), bierzemy z Tier B.
    if current_count < target_pool_size:
        needed = target_pool_size - current_count

        # Gdy Tier B jest mały, bierzemy całość
        if len(tier_b) <= needed:
            working_parts.append(tier_b)
            current_count += len(tier_b)
        # Gdy tier B jest duży, musimy wybierać.
        else:
            # Slot dla wysokich popularity
            n_pop = int(needed * popularity_rescue_ratio)
            # Slot dla najlepszego dopasowania semantycznego (Score)
            n_score = needed - n_pop

            # Wybieramy najpopularniejsze z dostępnych w Tier B
            b_pop_rescued = tier_b.sort_values("popularity", ascending=False).head(n_pop)
            # Usuwamy wybrane hity z puli, aby ich nie zduplikować
            remaining_b = tier_b.drop(b_pop_rescued.index)
            # Dobieramy resztę na podstawie czystego wyniku (score)
            b_score_top = remaining_b.sort_values("score", ascending=False).head(n_score)

            working_parts.extend([b_pop_rescued, b_score_top])
            current_count += (len(b_pop_rescued) + len(b_score_top))

    # Sprawdzamy, czy osiągnęliśmy absolutne minimum wymagane przez użytkownika (np. 15 piosenek na playlistę).
    # Jeśli nie, musimy sięgnąć po "słabsze" utwory, bo lepiej dać cokolwiek niż pustą playlistę.
    if current_count < min_required_size:
        needed = min_required_size - current_count
        part_c = tier_c.sort_values("score", ascending=False).head(needed)
        working_parts.append(part_c)

    # Zabezpieczenie na wypadek pustych wszystkich Tierów
    if not working_parts:
        return pd.DataFrame(columns=tier_a.columns)

    # Sklejamy
    working = pd.concat(working_parts, ignore_index=True)

    # Ustalamy po czym identyfikujemy piosenkę.
    subset_cols = []

    if 'id' in working.columns:
        subset_cols = ['id']
    elif 'name' in working.columns and 'artist' in working.columns:
        subset_cols = ['name', 'artist']

    if subset_cols:
        working = working.drop_duplicates(subset=subset_cols)
    else:
        pass

    # Finalne sortowanie
    working = working.sort_values("score", ascending=False).reset_index(drop=True)

    return working




working = build_working_set(
    tier_a, tier_b, tier_c,
    target_pool_size=WORKSET_CONFIG['target_pool_size'],
    min_required_size=WORKSET_CONFIG['min_required_size'],
    popularity_rescue_ratio=WORKSET_CONFIG['popularity_rescue_ratio']
)
# working
working["score"].value_counts()




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




pop_high, pop_mid, pop_low = bucket_by_popularity(working, POPULARITY_CONFIG['p_high'], POPULARITY_CONFIG['p_mid'])
pop_high





def weighted_sample(df: pd.DataFrame, k: int, alpha: float):
    """
    Losuje `k` wierszy z DataFrame bez zwracania, faworyzując utwory z wyższym 'score'.
    Prawdopodobieństwo wylosowania utworu jest proporcjonalne do: (score ^ alpha).

    Rola parametru alpha (temperatura losowania):
    - alpha > 1 (np. 3.0): Strategia "Greedy" (chciwa). Bardzo mocno faworyzuje utwory z najwyższym wynikiem.
      Szansa na wylosowanie słabszych utworów drastycznie maleje.
    - alpha = 1.0: Prawdopodobieństwo liniowo zależne od wyniku.
    - alpha < 1 (np. 0.5): Spłaszczenie różnic. Słabsze utwory mają większą szansę zaistnieć.
    - alpha = 0: Losowanie jednostajne (każdy utwór ma taką samą szansę, ignorujemy score).

    Args:
        df (pd.DataFrame): Tabela z kandydatami (musi zawierać kolumnę 'score').
        k (int): Liczba utworów do wylosowania.
        alpha (float): Współczynnik skalowania wag (wykładnik potęgi).

    Returns:
        pd.DataFrame: Podzbiór `df` zawierający `k` wylosowanych wierszy.
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
    Dokonuje finalnego wyboru utworów do playlisty z przygotowanej "puli roboczej" (working set).
    Stosuje zaawansowaną strategię mieszania popularności (Popularity Mix) oraz ważonego losowania (Weighted Sampling).
    Gdy pożądanych utworów jest mała (5), to wykonuje losowanie bez podziału na buckety.

    Strategia:
    1. Forced Popular (Kotwice): Gwarantuje obecność znanych hitów (np. 2-3 utwory), aby użytkownik czuł się bezpiecznie.
       Stosuje losowanie ważone, aby nie grać w kółko tych samych 5 najpopularniejszych piosenek
    2. Popularity Buckets: Dzieli resztę miejsc wg proporcji (np. 40% High, 40% Mid, 20% Low/Discovery).
    3. Weighted Audio/Tag Score: Wewnątrz każdego bucketa utwory są losowane z prawdopodobieństwem zależnym od ich dopasowania
       do zapytania (score). Dzięki temu playlista trzyma klimat (audio/tagi), ale jest różnorodna.

    Args:
        working (pd.DataFrame): Pula kandydatów (zbiór roboczy) z kolumnami 'score' i 'popularity'.
        popularity_cfg (Dict[str, Any]): Konfiguracja popularności.
            - 'p_high', 'p_mid' (int): Progi punktowe podziału na buckety.
            - 'mix' (dict): Proporcje (np. {'high': 0.4, 'mid': 0.4, 'low': 0.2}).
            - 'forced_popular' (int): Liczba gwarantowanych hitów na start.
            - 'forced_popular_min' (int): Minimalna popularność, by utwór uznano za "hit" do sekcji forced.
        sampling_cfg (Dict[str, Any]): Konfiguracja losowania.
            - 'final_n' (int): Docelowa długość playlisty.
            - 'alpha' (float): "Temperatura" losowania. Im wyższa, tym bardziej faworyzujemy utwory z wysokim score.
            - 'shuffle' (bool): Czy przetasować finalną listę (True) czy zostawić posortowaną po score (False/Debug).

    Returns:
        pd.DataFrame: Gotowa playlista o długości `final_n` (lub mniejszej, jeśli brakło kandydatów).
    """

    if len(working) == 0:
        return working.iloc[0:0].copy()

    # Konfiguracja
    final_n = sampling_cfg.get("final_n", 15)
    alpha   = sampling_cfg.get("alpha", 2.0) # Siła wpływu score na losowanie

    # Gdy chcemy mało piosenek, po prostu zwracamy losowe kilka bez podziału na buckety
    if final_n < 5:
        return weighted_sample(working, final_n, alpha)

    p_high = popularity_cfg.get("p_high", 70)
    p_mid  = popularity_cfg.get("p_mid", 35)
    mix    = popularity_cfg.get("mix", {"high": 0.4, "mid": 0.35, "low": 0.25})

    forced_popular         = popularity_cfg.get("forced_popular", 0)
    forced_popular_min     = popularity_cfg.get("forced_popular_min", p_high)

    # 1. Bucketowanie po popularności
    pop_high, pop_mid, pop_low = bucket_by_popularity(working, p_high=p_high, p_mid=p_mid)

    final_parts = []

    # 2. Forced popular
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

    # 3. Obliczamy ile slotów zostało do wypełnienia
    n_forced = len(forced_taken)
    remaining = max(0, final_n - n_forced)

    if remaining == 0:
        combined = pd.concat(final_parts, ignore_index=False)
        return combined.sort_values("score", ascending=False).reset_index(drop=True)

    # 4. Wyliczamy cele dla bucketów (Mix)
    target_high = int(round(remaining * mix.get("high", 0.0)))
    target_mid  = int(round(remaining * mix.get("mid",  0.0)))
    target_low  = remaining - target_high - target_mid  # Reszta do low

    # Clamp (nie możemy wziąć więcej niż jest w buckecie)
    target_high = min(target_high, len(pop_high))
    target_mid  = min(target_mid, len(pop_mid))
    target_low  = min(target_low, len(pop_low))

    # 5. Główne losowanie z bucketów (Weighted Sample wg Score)
    # Piosenki, które lepiej pasują (wyższy score), mają większą szansę na wylosowanie.
    sampled_high = weighted_sample(pop_high, target_high, alpha)
    sampled_mid  = weighted_sample(pop_mid,  target_mid,  alpha)
    sampled_low  = weighted_sample(pop_low,  target_low,  alpha)

    final_parts.extend([sampled_high, sampled_mid, sampled_low])

    combined = pd.concat(final_parts, ignore_index=False)

    # 6. Fill Gaps (Jeśli zaokrąglenia albo braki w bucketach sprawiły, że mamy za mało)
    if len(combined) < final_n:
        missing = final_n - len(combined)
        used_idx = set(combined.index)

        # Bierzemy resztki z całego working setu
        remaining_pool = working[~working.index.isin(used_idx)]
        extra = weighted_sample(remaining_pool, missing, alpha)

        combined = pd.concat([combined, extra], ignore_index=False)

    # 6. finalne tasowanie
    # frac=1 - "weź 100% wierszy, ale w losowej kolejności"
    do_shuffle = sampling_cfg.get("shuffle", True)
    if do_shuffle:
        combined = combined.sample(frac=1).reset_index(drop=True)
    else:
        # do debugowania: najlepsze na górze
        combined = combined.sort_values("score", ascending=False).reset_index(drop=True)

    return combined




picked_prompt



sample_final_songs(
    working,
    POPULARITY_CONFIG,
    SAMPLING_CONFIG
)[["name", "artist", "tags_list", "popularity", "score", "n_tempo", "energy", "valence", "instrumentalness", "n_loudness"]]