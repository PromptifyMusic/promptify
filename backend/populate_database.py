# -*- coding: utf-8 -*-
"""
Skrypt do wstawienia przykładowych danych do tabeli spotify_tracks
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Dane testowe
sample_tracks = [
    {
        'track_id': 'TRAAADT12903CCC339',
        'name': 'Andalucia',
        'artist': 'Pink Martini',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/37d9c700532305fc07e22900a55d2c6fbe1d3858',
        'tags': 'jazz, french, lounge',
        'genre': None,
        'year': 1997,
        'duration_ms': 219360,
        'danceability': 0.714,
        'energy': 0.521,
        'key': 2,
        'loudness': -9.828,
        'mode': 1,
        'speechiness': 0.0363,
        'acousticness': 0.727,
        'instrumentalness': 0.825,
        'liveness': 0.074,
        'valence': 0.683,
        'tempo': 92.53,
        'time_signature': 3,
        'album_name': 'Sympathique',
        'popularity': 35,
        'spotify_url': 'https://open.spotify.com/track/4A5pkIW4nXoS4lUy9nu4P6',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b273b31bd144093d6e0ad94318a8", "width": 640, "height": 640}]',
        'spotify_id': '4A5pkIW4nXoS4lUy9nu4P6',
        'n_tempo': 0.387,
        'n_loudness': 0.788,
        'tags_list': "['jazz', 'french', 'lounge']",
        'tags_count': 3
    },
    {
        'track_id': 'TRAAAED128E0783FAB',
        'name': "It's About Time",
        'artist': 'Jamie Cullum',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/9bb7b58f848a742b0a8ec3fd538e8c8970877348',
        'tags': 'jazz, piano, chill',
        'genre': 'Jazz',
        'year': 2004,
        'duration_ms': 247000,
        'danceability': 0.565,
        'energy': 0.507,
        'key': 9,
        'loudness': -8.339,
        'mode': 1,
        'speechiness': 0.0541,
        'acousticness': 0.702,
        'instrumentalness': 0.0000664,
        'liveness': 0.129,
        'valence': 0.454,
        'tempo': 152.462,
        'time_signature': 4,
        'album_name': 'Twentysomething',
        'popularity': 34,
        'spotify_url': 'https://open.spotify.com/track/79AtmE4M8aPPYt5v03atAp',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b2736108cbae948623a53e060833", "width": 640, "height": 640}]',
        'spotify_id': '79AtmE4M8aPPYt5v03atAp',
        'n_tempo': 0.638,
        'n_loudness': 0.812,
        'tags_list': "['jazz', 'piano', 'chill']",
        'tags_count': 3
    },
    {
        'track_id': 'TRAAAHD128F42635A5',
        'name': "I'll Be Waiting",
        'artist': 'Adele',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/5f1c0da2922b0952bb45953ff43076a1cdb95f48',
        'tags': 'pop, female_vocalists, singer_songwriter, british, soul',
        'genre': None,
        'year': 2011,
        'duration_ms': 241653,
        'danceability': 0.548,
        'energy': 0.843,
        'key': 2,
        'loudness': -2.674,
        'mode': 1,
        'speechiness': 0.0374,
        'acousticness': 0.0467,
        'instrumentalness': 0,
        'liveness': 0.131,
        'valence': 0.754,
        'tempo': 160.009,
        'time_signature': 4,
        'album_name': '21',
        'popularity': 54,
        'spotify_url': 'https://open.spotify.com/track/42Uw9frN5ZVX40mRU3hHFL',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b273164feb363334f93b6458d2a9", "width": 640, "height": 640}]',
        'spotify_id': '42Uw9frN5ZVX40mRU3hHFL',
        'n_tempo': 0.670,
        'n_loudness': 0.901,
        'tags_list': "['pop', 'female_vocalists', 'singer_songwriter', 'british', 'soul']",
        'tags_count': 5
    },
    {
        'track_id': 'TRAAAQO12903CD8E1C',
        'name': 'Take Time',
        'artist': 'The Books',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/1de501d7c5968bcad0a4d473292ec1d87a04de9e',
        'tags': 'electronic, indie, experimental',
        'genre': None,
        'year': 2003,
        'duration_ms': 216626,
        'danceability': 0.72,
        'energy': 0.643,
        'key': 10,
        'loudness': -7.805,
        'mode': 1,
        'speechiness': 0.1,
        'acousticness': 0.58,
        'instrumentalness': 0.0728,
        'liveness': 0.345,
        'valence': 0.504,
        'tempo': 90.037,
        'time_signature': 4,
        'album_name': 'The Lemon of Pink (Remastered)',
        'popularity': 25,
        'spotify_url': 'https://open.spotify.com/track/3NXOn6Jurm7LXk8P2S3aBj',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b27380ae246b983f32fb58e89194", "width": 640, "height": 640}]',
        'spotify_id': '3NXOn6Jurm7LXk8P2S3aBj',
        'n_tempo': 0.377,
        'n_loudness': 0.820,
        'tags_list': "['electronic', 'indie', 'experimental']",
        'tags_count': 3
    },
    {
        'track_id': 'TRAAAZF12903CCCF6B',
        'name': 'Break My Stride',
        'artist': 'Matthew Wilder',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/06ba33dcbe01b55115404c7b1436d689833da424',
        'tags': 'pop, 80s, new_wave',
        'genre': None,
        'year': 2009,
        'duration_ms': 182480,
        'danceability': 0.908,
        'energy': 0.695,
        'key': 2,
        'loudness': -7.68,
        'mode': 0,
        'speechiness': 0.0448,
        'acousticness': 0.118,
        'instrumentalness': 0,
        'liveness': 0.126,
        'valence': 0.871,
        'tempo': 109.664,
        'time_signature': 4,
        'album_name': "I Don't Speak The Language",
        'popularity': 74,
        'spotify_url': 'https://open.spotify.com/track/1mCsF9Tw4AkIZOjvZbZZdT',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b2739824c6e084b02d24b2e22e94", "width": 640, "height": 640}]',
        'spotify_id': '1mCsF9Tw4AkIZOjvZbZZdT',
        'n_tempo': 0.459,
        'n_loudness': 0.822,
        'tags_list': "['pop', '80s', 'new_wave']",
        'tags_count': 3
    },
    {
        'track_id': 'TRAABJS128F9325C99',
        'name': 'Auburn and Ivory',
        'artist': 'Beach House',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/b3583a8703729f6a8d3ec5620c7c25cd257448ec',
        'tags': 'indie, downtempo',
        'genre': 'Rock',
        'year': 2006,
        'duration_ms': 270026,
        'danceability': 0.438,
        'energy': 0.285,
        'key': 5,
        'loudness': -14.96,
        'mode': 0,
        'speechiness': 0.0265,
        'acousticness': 0.866,
        'instrumentalness': 0.00000288,
        'liveness': 0.0696,
        'valence': 0.366,
        'tempo': 145.601,
        'time_signature': 3,
        'album_name': 'Beach House',
        'popularity': 39,
        'spotify_url': 'https://open.spotify.com/track/5cqaG09jwHAyDURuZXViwC',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b27369d8e3478a4964cf7219f3da", "width": 640, "height": 640}]',
        'spotify_id': '5cqaG09jwHAyDURuZXViwC',
        'n_tempo': 0.609,
        'n_loudness': 0.708,
        'tags_list': "['indie', 'downtempo']",
        'tags_count': 2
    },
    {
        'track_id': 'TRAABXA128F9326351',
        'name': 'Brain',
        'artist': 'Banks',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/80b00112c3e5ed4d251edbba12ffa5a06c9670da',
        'tags': 'electronic, pop, female_vocalists, dance, chillout',
        'genre': None,
        'year': 2014,
        'duration_ms': 281640,
        'danceability': 0.306,
        'energy': 0.498,
        'key': 4,
        'loudness': -8.392,
        'mode': 0,
        'speechiness': 0.0374,
        'acousticness': 0.253,
        'instrumentalness': 0.035,
        'liveness': 0.188,
        'valence': 0.0807,
        'tempo': 90.326,
        'time_signature': 4,
        'album_name': 'Goddess (Deluxe)',
        'popularity': 45,
        'spotify_url': 'https://open.spotify.com/track/4dDoIid58lgImNuYAxTRyM',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b2739c8dd74225a1fb838fa7dca6", "width": 640, "height": 640}]',
        'spotify_id': '4dDoIid58lgImNuYAxTRyM',
        'n_tempo': 0.378,
        'n_loudness': 0.811,
        'tags_list': "['electronic', 'pop', 'female_vocalists', 'dance', 'chillout']",
        'tags_count': 5
    },
    {
        'track_id': 'TRAACCD128F422CDA8',
        'name': 'Just a Girl',
        'artist': 'No Doubt',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/78232b764e4b2406c160020e9f422f7d8c111bfa',
        'tags': 'rock, alternative, pop, female_vocalists, punk, 90s',
        'genre': None,
        'year': 2019,
        'duration_ms': 209986,
        'danceability': 0.639,
        'energy': 0.911,
        'key': 2,
        'loudness': -4.599,
        'mode': 1,
        'speechiness': 0.0496,
        'acousticness': 0.0848,
        'instrumentalness': 0.0000576,
        'liveness': 0.148,
        'valence': 0.748,
        'tempo': 108.004,
        'time_signature': 4,
        'album_name': 'Tragic Kingdom',
        'popularity': 74,
        'spotify_url': 'https://open.spotify.com/track/5lWRaa0fBxDE5yU91npPq7',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b2736ebd5e789646a833b8f7d4ba", "width": 640, "height": 640}]',
        'spotify_id': '5lWRaa0fBxDE5yU91npPq7',
        'n_tempo': 0.452,
        'n_loudness': 0.871,
        'tags_list': "['rock', 'alternative', 'pop', 'female_vocalists', 'punk', '90s']",
        'tags_count': 6
    },
    {
        'track_id': 'TRAACQW128F428854F',
        'name': 'The Prettiest Star',
        'artist': 'David Bowie',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/9e995d4846b049c1b64ce4efb4ee302db98224d1',
        'tags': 'rock, classic_rock, singer_songwriter, 70s, male_vocalists',
        'genre': None,
        'year': 2005,
        'duration_ms': 189346,
        'danceability': 0.344,
        'energy': 0.53,
        'key': 5,
        'loudness': -6.154,
        'mode': 1,
        'speechiness': 0.0296,
        'acousticness': 0.175,
        'instrumentalness': 0.0885,
        'liveness': 0.333,
        'valence': 0.539,
        'tempo': 109.469,
        'time_signature': 4,
        'album_name': 'Aladdin Sane (2013 Remaster)',
        'popularity': 48,
        'spotify_url': 'https://open.spotify.com/track/2tjskutXjrrfAyoSw4q4kb',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b2735db6dbaca8678527e643a866", "width": 640, "height": 640}]',
        'spotify_id': '2tjskutXjrrfAyoSw4q4kb',
        'n_tempo': 0.458,
        'n_loudness': 0.846,
        'tags_list': "['rock', 'classic_rock', 'singer_songwriter', '70s', 'male_vocalists']",
        'tags_count': 5
    },
    {
        'track_id': 'TRAACZN128F93236B1',
        'name': 'Seaweed',
        'artist': 'Skalpel',
        'spotify_preview_url': 'https://p.scdn.co/mp3-preview/a1abb838b5a1b59dcd5b340b58c56760aff45b98',
        'tags': 'jazz, downtempo, lounge',
        'genre': 'Rap',
        'year': 2005,
        'duration_ms': 150546,
        'danceability': 0.594,
        'energy': 0.405,
        'key': 2,
        'loudness': -9.674,
        'mode': 1,
        'speechiness': 0.0356,
        'acousticness': 0.339,
        'instrumentalness': 0.654,
        'liveness': 0.26,
        'valence': 0.0697,
        'tempo': 115.6,
        'time_signature': 3,
        'album_name': 'Konfusion',
        'popularity': 11,
        'spotify_url': 'https://open.spotify.com/track/5TVDgWERU2w2hsJlTq0pXD',
        'explicit': False,
        'album_images': '[{"url": "https://i.scdn.co/image/ab67616d0000b27373bece81d0786fc1f7bb80f4", "width": 640, "height": 640}]',
        'spotify_id': '5TVDgWERU2w2hsJlTq0pXD',
        'n_tempo': 0.484,
        'n_loudness': 0.791,
        'tags_list': "['jazz', 'downtempo', 'lounge']",
        'tags_count': 3
    }
]

def insert_sample_data():
    """Wstawia przykładowe dane do tabeli spotify_tracks"""

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[BLAD] Brak DATABASE_URL w pliku .env")
        return False

    try:
        engine = create_engine(database_url, pool_pre_ping=True)

        with engine.connect() as conn:
            # Sprawdź czy tabela jest pusta
            result = conn.execute(text("SELECT COUNT(*) FROM spotify_tracks"))
            count = result.fetchone()[0]
            print(f"[INFO] Aktualna liczba rekordow w tabeli: {count}")

            # Wstaw dane
            inserted = 0
            for track in sample_tracks:
                try:
                    # Konwertuj album_images na poprawny format JSONB
                    album_images_str = track['album_images'].replace("'", '"')

                    query = text("""
                        INSERT INTO spotify_tracks (
                            track_id, name, artist, spotify_preview_url, tags, genre, year,
                            duration_ms, danceability, energy, key, loudness, mode, speechiness,
                            acousticness, instrumentalness, liveness, valence, tempo, time_signature,
                            album_name, popularity, spotify_url, explicit, album_images, spotify_id,
                            n_tempo, n_loudness, tags_list, tags_count
                        ) VALUES (
                            :track_id, :name, :artist, :spotify_preview_url, :tags, :genre, :year,
                            :duration_ms, :danceability, :energy, :key, :loudness, :mode, :speechiness,
                            :acousticness, :instrumentalness, :liveness, :valence, :tempo, :time_signature,
                            :album_name, :popularity, :spotify_url, :explicit,
                            CAST(:album_images AS jsonb), :spotify_id,
                            :n_tempo, :n_loudness, :tags_list, :tags_count
                        )
                        ON CONFLICT (track_id) DO NOTHING
                    """)

                    params = track.copy()
                    params['album_images'] = album_images_str

                    conn.execute(query, params)
                    inserted += 1
                    print(f"[OK] Wstawiono: {track['name']} - {track['artist']}")
                except Exception as e:
                    print(f"[BLAD] Nie udalo sie wstawic {track['name']}: {str(e)[:200]}")

            conn.commit()

            # Sprawdź końcową liczbę rekordów
            result = conn.execute(text("SELECT COUNT(*) FROM spotify_tracks"))
            final_count = result.fetchone()[0]
            print(f"\n[INFO] Koncowa liczba rekordow: {final_count}")
            print(f"[INFO] Wstawiono nowych rekordow: {inserted}")

            # Pokaż przykładowe dane
            print("\n[INFO] Przykladowe utwory w bazie:")
            result = conn.execute(text("SELECT track_id, name, artist FROM spotify_tracks LIMIT 5"))
            for row in result:
                print(f"  - {row[1]} by {row[2]} (ID: {row[0]})")

            return True

    except Exception as e:
        print(f"[BLAD] Blad polaczenia z baza danych:")
        print(f"  {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("WSTAWIANIE DANYCH DO TABELI spotify_tracks")
    print("=" * 60)

    success = insert_sample_data()

    print("\n" + "=" * 60)
    if success:
        print("[OK] Dane zostaly pomyslnie wstawione!")
    else:
        print("[BLAD] Wystapil problem podczas wstawiania danych")
    print("=" * 60)

