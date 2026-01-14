
EXTRACTION_CONFIG = {
    "gliner_threshold": 0.15,            # Min. pewność klasyfikacji encji GLiNER
    "tag_similarity_threshold": 0.82,   # Min. cosine similarity dla mapowania na tagi (E5)
    "tag_similarity_threshold_lenient": 0.75, # Min. cosine similarity dla mapowania na tagi (E5) w przypadku, gdy fraza zawiera tag
    "audio_confidence_threshold": 0.78, # Min. pewność mapowania na cechy audio
    "fuzzy_cutoff": 0.9 # fuzzy match próg
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
    "min_required_size": 15,

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


GENRE_MODIFIERS = [
    # eng
    "post", "hard", "heavy", "soft", "pop", "alt", "alternative", "indie", 
    "nu", "neo", "progressive", "psychedelic", "acid", "experimental", 
    "symphonic", "death", "black", "thrash", "doom", "power", "speed", 
    "glam", "gothic", "tech", "deep", "classic", "modern", "electronic", 
    "industrial", "melodic", "folk", "lo-fi", "lofi", "ambient", "synth",
    "electro", "acoustic", "instrumental", "gangsta", "brit", "new", "dark",
    
    # pl
    "alternatywny", "progresywny", "psychodeliczny", "symfoniczny", 
    "ciężki", "klasyczny", "elektroniczny", "gotycki", "melodyjny", 
    "eksperymentalny", "współczesny", "industrialny", "akustyczny",
    "instrumentalny", "polski", "zagraniczny"
]

LABELS_CONFIG = {
    # --- KATEGORIE KIEROWANE DO BAZY TAGÓW (route: TAGS) ---

    "music_genre": {
        "desc": "music genre, gatunek muzyczny, category, style, \
        rock, pop, jazz, hip hop, rap, rapowa, metal, indie, alternative, \
        electronic, classical, reggae, techno, house, folk, blues, \
        funk, soul, punk, grunge, dubstep, dnb, disco, k-pop, \
        r&b, experimental, eksperymentalna, lo-fi, ambient",
        "route": "TAGS"
    },

    "song_type": {
        "desc": "soundtrack, ost, film score, remix, cover, \
        live performance, concert, instrumental, acoustic, unplugged, \
        opening, ending, wersja instrumentalna, na żywo",
        "route": "TAGS"
    },

    "musical_instrument": {
        "desc": "instrument, piano, pianino, guitar, gitara, drums, perkusja, \
        violin, skrzypce, bass, bas, saxophone, saksofon, synthesizer, syntezator",
        "route": "TAGS"
    },

    "vocal_characteristics": {
        "desc": "vocals, voice type, female vocalist, wokalistka, \
        male vocalist, wokalista, męski głos, kobiecy głos, choir, chór",
        "route": "TAGS"
    },

    "geographical_location": {
        "desc": "country, origin, language, polish, polska, polski, \
        american, usa, british, uk, french, francuska, german, spanish, \
        russian, japanese, korean",
        "route": "TAGS"
    },

    "time_period": {
        "desc": "era, decade, year, 80s, 90s, 00s, 2020s, 70s, 60s, \
        lata 80, lata 90, lata 70, oldies, retro, klasyk",
        "route": "TAGS"
    },

    # --- DANE AUDIO (route: AUDIO) ---

    # "audio_characteristics": {
    #     "desc": "mood, emotion, tempo, vibe, atmosphere, \
    #     sad, smutna, melancholic, happy, wesoła, joyful, \
    #     fast, szybka, dynamic, energetic, slow, wolna, calm, spokojna, \
    #     relaxing, relaksująca, loud, głośna, quiet, cicha, \
    #     heavy, ciężka, soft, delikatna, danceable, taneczna, rhythmic",
    #     "route": "AUDIO"
    # }
    "audio_emotion": {
        "desc": "emotion, feelings, uczucia, \
        happy, wesoła, wesoły, joyful, radosna, positive, pozytywna, \
        sad, smutna, smutny, depressive, depresyjna, crying, płaczliwa, \
        angry, agresywna, zła, fear, straszna, emotional, uczuciowa, \
        euphoric, euforyczna",
        "route": "AUDIO"
    },


    # 2. NASTRÓJ / ATMOSFERA (Jaki jest klimat otoczenia/tła?)
    "audio_mood": {
        "desc": "mood, vibe, atmosphere, klimat, \
        chill, chillout, relaksująca, relaxing, calm, spokojna, peaceful, \
        dark, mroczna, gloomy, ponura, \
        romantic, romantyczna, love, miłosna, sensual, zmysłowa, \
        energetic, energetyczna, dance, taniec, party, imprezowa, epic, epicka, \
        dreamy, marzycielska, mysterious, tajemnicza",
        "route": "AUDIO"
    },

    # 3. CECHY TECHNICZNE / FIZYCZNE (Tempo, głośność, rytmika)
    "audio_technical": {
        "desc": "tempo, speed, volume, rhythm, \
        fast, szybka, szybkie tempo, high bpm, \
        slow, wolna, wolne tempo, downtempo, \
        dynamic, dynamiczna, rhythmic, rytmiczna, groove, \
        loud, głośna, noisy, hałaśliwa, \
        quiet, cicha, soft, delikatna, gentle, łagodna, \
        heavy, ciężka, mocna, hard, \
        acoustic, akustyczna, synthesized, syntetyczna",
        "route": "AUDIO"
    },

    "usage_context": {
        "desc": "activity, occasion, purpose, \
        focus, study, nauka, reading, coding, work, praca, \
        sleep, sen, relax, relaks, yoga, meditation, medytacja, \
        workout, gym, siłownia, running, bieganie, sport, training, \
        car, driving, auto, podróż, commute, spacer, \
        party, impreza, club, klub, dancing, taniec, \
        cleaning, sprzątanie, cooking, gotowanie, background, tło, \
        gaming, gry, playing, granie, stream, \
        date, randka, romance, miłość, sex, seks",
        "route": "AUDIO"
    },
}
#dodane
LANGUAGE_CONFIG = {
    "polish":   ["polski", "polska", "poland", "pl", "rodzimy", "krajowy", "polskie", "polsku"],
    "british": ["brytyjski", "brytania", "anglia", "londyn", "szkocki", "walijski","british", "uk", "england", "brit", "scotland", "wales", "london"],
    "american": ["amerykański", "ameryka", "usa", "stany", "zjednoczone","american", "america", "us", "states"],
    "english": ["angielski", "english", "anglojęzyczny", "angielsku"],
    "german":   ["niemiecki", "niemcy", "german", "deutsch", "germany", "niemiecku"],
    "french":   ["francuski", "francja", "french", "france", "francusku"],
    "spanish":  ["hiszpański", "hiszpania", "spanish", "latino", "latynoski", "spain", "hiszpańsku"],
    "italian":  ["włoski", "włochy", "italia", "italian", "italy", "włosku"],
    "russian":  ["rosyjski", "rosja", "ruski", "moskiewski", "russian", "russia", "rosyjsku"],
    "korean":   ["koreański", "korea", "korean", "koreańsku"],
    "japanese": ["japoński", "japonia", "japan", "anime", "japanese", "japońsku"],
    "swedish":  ["szwedzki", "szwecja", "swedish", "sweden", "szwedzku"],
    "foreign": ["zagraniczny", "obcy", "międzynarodowy", "światowy", "foreign", "international", "overseas","global"],
}

GENRE_PHRASES_EXACT = {
    "rap": ["hip hop", "hip-hop", "hiphop", "new school", "old school"],
    "electronic": ["drum and bass", "dnb", "drum & bass", "psy trance"],
    "rock": ["post rock", "post-rock", "alt rock", "punk rock"],
    "pop": ["k-pop", "kpop", "synth pop"],
    "rnb": ["r&b", "rnb"]
}

GENRE_LEMMA_CONFIG = {
    "rap": ["rap", "rapowy", "rapsy", "trap", "drill"],
    "rock": ["rock", "rockowy", "metal", "metalowy", "grunge", "grungowy", "punk", "punkowy", "alt"],
    "pop": ["pop", "popowy", "disco", "mainstream", "radiowy", "przebój", "hit"],
    "electronic": ["elektroniczny", "elektro", "techno", "house", "trance", "dubstep", "edm", "klubowy"],
    "jazz": ["jazz", "jazzowy", "dżez", "blues", "bluesowy", "funk", "funkowy", "soul", "soulowy"],
    "classical": ["klasyczny", "orkiestrowy", "symfoniczny", "classical", "ochestra", "symphonic"],
    "reggae": ["reggae", "rasta", "ska", "dub"],
    "folk": ["folk", "folkowy", "etno", "ludowy", "country"],
    "indie": ["indie", "alternatywny"]
}

LEMMA_TO_GENRE_MAP = {}
LEMMA_TO_TAG_MAP = {}



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
