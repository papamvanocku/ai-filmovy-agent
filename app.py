import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import json
import os
from google import genai
from google.genai import types

# ==========================================
# LOKÁLNÍ API KLÍČ (Pro hosting nech prázdné!)
# ==========================================
LOKALNI_API_KEY = ""  # <-- Na GitHub nahrávej PRÁZDNÉ! Na webu se klíč načte ze Secrets.

# Nastavení vzhledu stránky
st.set_page_config(page_title="AI Filmový Agent", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem !important;
    }
    div.stRadio > div[role="radiogroup"] {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# VÝBĚR JAZYKA A NADPIS
# ==========================================
col_nadpis, col_jazyk = st.columns([15, 1])

with col_jazyk:
    zvoleny_jazyk = st.selectbox(
        "Jazyk", 
        ["🇨🇿 CZ", "🇬🇧 EN"], 
        label_visibility="collapsed"
    )

lang = "CZ" if "CZ" in zvoleny_jazyk else "EN"

t = {
    "CZ": {
        "main_title": "🎬 AI Filmový Agent & Databáze",
        "nav_label": "Zobrazení:",
        "nav_chat": "💬 AI Filmový Agent",
        "nav_cat": "📚 Katalog všech filmů",
        "chat_welcome": "Ahoj! 👋 Jsem tvůj filmový AI agent. Napiš mi volně svými slovy, na co máš chuť, z jakého roku film hledáš, nebo jakou máš náladu (např. *'Chci akční filmy z roku 2023'* nebo *'Líbí se mi filmy s upíry'*), a já prohledám databázi!",
        "chat_placeholder": "Napiš zprávu AI agentovi...",
        "db_footer": "Databáze obsahuje nejnovější film ze dne 29. 9. 2023.",
        "missing_poster": "*(Plakát chybí)*",
        "detail": "Detail filmu",
        "year": "Rok vydání:",
        "genre": "Žánr:",
        "desc": "Popis:",
        "unknown": "Neznámý",
        "cat_title": "Prohlížení celé databáze (od roku 1995)",
        "sort_label": "Vyberte, jak chcete filmy seřadit:",
        "sort_az": "Abecedně (A-Z)",
        "sort_year": "Roku vydání (od nejnovějších)",
        "sort_genre": "Žánru",
        "col_title": "Název filmu",
        "col_year": "Datum vydání",
        "col_genre": "Žánry",
        "col_desc": "Popis děje",
        "loading": "Načítám filmovou databázi...",
        "thinking": "AI Agent přemýšlí a prohledává databázi...",
        "clear_chat": "🗑️ Vymazat chat",
        "no_movies_year": "Bohužel jsem pro zadaný rok a kritéria nenašel v databázi žádné filmy."
    },
    "EN": {
        "main_title": "🎬 AI Movie Agent & Database",
        "nav_label": "View:",
        "nav_chat": "💬 AI Movie Agent",
        "nav_cat": "📚 Full Movie Catalog",
        "chat_welcome": "Hi! 👋 I am your movie AI agent. Tell me in your own words what you're in the mood for, what release year you want, or what movies you like (e.g., *'Action movies from 2023'* or *'I like vampire movies'*), and I'll search the database!",
        "chat_placeholder": "Message the AI agent...",
        "db_footer": "The database contains movies up to September 29, 2023.",
        "missing_poster": "*(Poster missing)*",
        "detail": "Movie details",
        "year": "Release year:",
        "genre": "Genre:",
        "desc": "Overview:",
        "unknown": "Unknown",
        "cat_title": "Browse the entire database (from 1995)",
        "sort_label": "Choose how you want to sort the movies:",
        "sort_az": "Alphabetically (A-Z)",
        "sort_year": "Release year (newest first)",
        "sort_genre": "By Genre",
        "col_title": "Movie Title",
        "col_year": "Release Date",
        "col_genre": "Genres",
        "col_desc": "Overview",
        "loading": "Loading movie database...",
        "thinking": "AI Agent is thinking and searching the database...",
        "clear_chat": "🗑️ Clear chat",
        "no_movies_year": "Sorry, I couldn't find any movies matching that year and criteria in the database."
    }
}
_ = t[lang]

with col_nadpis:
    st.markdown(f"<h1 style='text-align: center; margin-top: -15px;'>{_['main_title']}</h1>", unsafe_allow_html=True)

# ==========================================
# POSTRANNÍ PANEL PRO NASTAVENÍ A PŘEPÍNÁNÍ POHLEDU
# ==========================================
with st.sidebar:
    st.header("🧭 Navigace")
    aktivni_stranka = st.radio(
        _['nav_label'],
        [_['nav_chat'], _['nav_cat']],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.header("🧠 Nastavení AI Agenta")
    
    secret_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            secret_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if secret_key.strip() != "":
        api_key = secret_key.strip()
        st.success("AI Agent je aktivní (klíč načten ze serveru)!")
    elif LOKALNI_API_KEY.strip() != "":
        api_key = LOKALNI_API_KEY.strip()
        st.success("AI Agent je aktivní (lokální klíč)!")
    else:
        api_key = st.text_input(
            "Vlož svůj Gemini API klíč:", 
            type="password",
            placeholder="AIzaSy...",
            help="Získáš zdarma na aistudio.google.com"
        )
        if api_key:
            st.success("AI Agent je aktivní!")
        else:
            st.info("Bez API klíče běží aplikace v základním režimu hledání slov.")
            
    if aktivni_stranka == _['nav_chat']:
        st.divider()
        if st.button(_['clear_chat'], use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ==========================================
# NAČTENÍ DAT A VEKTORIZACE
# ==========================================
@st.cache_resource
def nacti_model_a_data_web():
    if os.path.exists('filmy_web.csv.gz'):
        df = pd.read_csv('filmy_web.csv.gz', compression='gzip')
    else:
        df_raw = pd.read_csv('TMDB_movie_dataset_v11.csv')
        df = df_raw[['title', 'overview', 'genres', 'keywords', 'poster_path', 'release_date', 'popularity', 'vote_count']].copy()
        df['datum_vydani'] = pd.to_datetime(df['release_date'], errors='coerce')
        dnes = pd.Timestamp.today()
        df = df[(df['datum_vydani'].dt.year >= 1995) & (df['datum_vydani'] <= dnes)]
        df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
        df = df[df['vote_count'] >= 50]
        zakazana_slova = r'(?i)\b(porn|erotic|erotica|sex|tits?|titties|cum|cumshots?|ejaculat\w*|dick|pussy|cock|slut|whore|milf|hentai|orgasm|xxx|nympho|bondage|bdsm|squirt|boobs|softcore|pink film|seduction|steamy|sensual|affair|temptation of)\b'
        df = df[~df['title'].str.contains(zakazana_slova, regex=True, na=False)]
        df = df[~df['genres'].str.contains(zakazana_slova, regex=True, na=False)]
        df = df[~df['keywords'].str.contains(zakazana_slova, regex=True, na=False)]
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
        df = df.sort_values(by='popularity', ascending=False).head(30000).reset_index(drop=True)

    df['datum_vydani'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['rok'] = df['datum_vydani'].dt.year.fillna(0).astype(int)
    
    df = df.fillna('')
    df['text_pro_ai'] = df['title'] + " " + df['overview'] + " " + df['genres'] + " " + df['keywords']
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=25000)
    tfidf_matrix = tfidf.fit_transform(df['text_pro_ai'])
    
    return df, tfidf, tfidf_matrix

with st.spinner(_['loading']):
    df, tfidf, tfidf_matrix = nacti_model_a_data_web()

# ==========================================
# MOZEK AI AGENTA (LLM + FILTR ROKU + VYHLEDÁVÁNÍ)
# ==========================================
def zeptej_se_agenta(uzivatelsky_dotaz, klic, jazyk):
    odpoved_agenta = ""
    hledana_fraze = uzivatelsky_dotaz
    rok_od = None
    rok_do = None

    if klic and klic.strip() != "":
        client = genai.Client(api_key=klic.strip())
        instrukce = f"""
        Jsi expertní filmový AI agent. Uživatel ti napíše volným stylem, na jaký film má chuť, z jakého roku film hledá nebo zmíní oblíbené filmy.
        Tvým úkolem je vrátit JSON objekt se čtyřmi poli:
        1. "odpoved": Přátelská, přirozená reakce v jazyce '{jazyk}' (CZ = česky, EN = anglicky), kde uživateli potvrdíš jeho přání a představíš mu vybrané filmy.
        2. "klicova_slova": Řetězec (string) s 10 až 15 anglickými klíčovými slovy, žánry a tématy popisujícími děj (např. 'action adventure superhero thriller'). Pokud uživatel nezmínil žánr ani téma (napsal jen rok, např. 'filmy z roku 2022'), nech toto pole prázdné "".
        3. "rok_od": Celé číslo (integer) nebo null. Pokud uživatel zmínil konkrétní rok (např. 2022), nastav rok_od na 2022. Pokud zmínil období (např. 'od roku 2020' nebo '90. léta'), nastav začátek období (2020 nebo 1995). Jinak null.
        4. "rok_do": Celé číslo (integer) nebo null. Pokud uživatel zmínil konkrétní rok (např. 2022), nastav rok_do na 2022. Pokud zmínil období (např. '90. léta'), nastav konec období (1999). Jinak null.
        """
        
        modely_k_vyzkouseni = ['gemini-3.5-flash-lite', 'gemini-3.5-flash', 'gemini-3.8-flash']
        uspech = False
        posledni_chyba = ""
        
        for model_name in modely_k_vyzkouseni:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=uzivatelsky_dotaz,
                    config=types.GenerateContentConfig(
                        system_instruction=instrukce,
                        response_mime_type="application/json",
                    ),
                )
                data = json.loads(response.text)
                odpoved_agenta = data.get("odpoved", "")
                hledana_fraze = data.get("klicova_slova", "")
                rok_od = data.get("rok_od")
                rok_do = data.get("rok_do")
                uspech = True
                break
            except Exception as e:
                posledni_chyba = str(e)
                continue
                
        if not uspech:
            odpoved_agenta = f"⚠️ Všechny LLM modely jsou momentálně přetížené (používám lokální vyhledávání): {posledni_chyba}"
            hledana_fraze = uzivatelsky_dotaz
    else:
        odpoved_agenta = (
            "Zde jsou filmy z databáze, které nejlépe odpovídají tvým slovům:"
            if jazyk == "CZ" else
            "Here are the closest matches from the database:"
        )

    if isinstance(hledana_fraze, list):
        hledana_fraze = " ".join(str(slovo) for slovo in hledana_fraze)
    hledana_fraze = str(hledana_fraze).strip()

    maska = pd.Series(True, index=df.index)
    if rok_od is not None:
        maska = maska & (df['rok'] >= int(rok_od))
    if rok_do is not None:
        maska = maska & (df['rok'] <= int(rok_do))
        
    platne_indexy = df[maska].index.to_numpy()
    
    if len(platne_indexy) == 0:
        return _['no_movies_year'], []

    if hledana_fraze == "":
        doporucene_indexy = platne_indexy[:5]
        return odpoved_agenta, doporucene_indexy

    uzivatelsky_vektor = tfidf.transform([hledana_fraze])
    sim_scores = cosine_similarity(uzivatelsky_vektor, tfidf_matrix[platne_indexy]).flatten()
    
    if sim_scores.max() == 0:
        doporucene_indexy = platne_indexy[:5]
    else:
        lokalni_top = sim_scores.argsort()[::-1][:5]
        doporucene_indexy = platne_indexy[lokalni_top]
    
    return odpoved_agenta, doporucene_indexy

def vykresli_filmy_v_chatu(movie_indices):
    if len(movie_indices) == 0:
        return
    cols = st.columns(5)
    base_url = "https://image.tmdb.org/t/p/w500"
    for col, index_filmu in zip(cols, movie_indices):
        film = df.iloc[index_filmu]
        with col:
            st.markdown(f"<p style='text-align: center; margin-bottom: 5px; font-size: 0.9rem;'><strong>{film['title']}</strong></p>", unsafe_allow_html=True)
            if film['poster_path']:
                st.image(base_url + film['poster_path'], use_container_width=True)
            else:
                st.write(_['missing_poster'])
            
            with st.expander(_['detail']):
                rok = film['release_date'][:4] if film['release_date'] else _['unknown']
                st.write(f"**{_['year']}** {rok}")
                st.write(f"**{_['genre']}** {film['genres']}")
                st.write(f"**{_['desc']}** {film['overview']}")

# ==========================================
# ZOBRAZENÍ PODLE VYBRANÉ STRÁNKY V MENU
# ==========================================
if aktivni_stranka == _['nav_chat']:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.chat_message("assistant"):
        st.markdown(_['chat_welcome'])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "movies" in msg:
                vykresli_filmy_v_chatu(msg["movies"])

    prompt = st.chat_input(_['chat_placeholder'])
    st.markdown(
        f"<p style='text-align: center; color: gray; font-size: 0.78rem; margin-top: -10px; margin-bottom: 0px;'>{_['db_footer']}</p>",
        unsafe_allow_html=True
    )

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner(_['thinking']):
            odpoved_text, movie_indices = zeptej_se_agenta(prompt, api_key, lang)

        with st.chat_message("assistant"):
            st.markdown(odpoved_text)
            vykresli_filmy_v_chatu(movie_indices)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": odpoved_text, 
            "movies": movie_indices
        })

else:
    st.markdown(f"<h3 style='text-align: center;'>{_['cat_title']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-weight: bold;'>{_['sort_label']}</p>", unsafe_allow_html=True)
    
    razeni = st.radio(
        "Skrytý popisek pro řazení",
        [_['sort_az'], _['sort_year'], _['sort_genre']],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    zobrazene_df = df[['title', 'release_date', 'genres', 'overview']].copy()
    zobrazene_df.columns = [_['col_title'], _['col_year'], _['col_genre'], _['col_desc']]
    
    if razeni == _['sort_az']:
        zobrazene_df = zobrazene_df.sort_values(by=_['col_title'])
    elif razeni == _['sort_year']:
        zobrazene_df = zobrazene_df.sort_values(by=_['col_year'], ascending=False)
    elif razeni == _['sort_genre']:
        zobrazene_df = zobrazene_df.sort_values(by=_['col_genre'])
        
    st.dataframe(zobrazene_df, use_container_width=True, height=600)
