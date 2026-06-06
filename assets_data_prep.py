import re
import ast
import numpy as np
import pandas as pd


def prepare_data(df):
    data = df.copy()
    # ─── הסרת עמודות אסורות (Data Leakage) ──────────────────────────────────
    # averageRating = משתנה המטרה, numVotes/BoxOffice = מידע עתידי
    drop_early = ['averageRating', 'numVotes', 'BoxOffice', 'tconst', 'primaryTitle']
    data = data.drop(columns=[c for c in drop_early if c in data.columns])
    # ─── שנת יציאה ───────────────────────────────────────────────────────────
    # שנים לפני 1900 הן שגיאות קלט — ממירים ל-NaN במקום למחוק (שומר על השורה)
    if 'startYear' in data.columns:
        data.loc[data['startYear'] < 1900, 'startYear'] = np.nan
    # ─── ניקוי ופירוק ז'אנרים ────────────────────────────────────────────────
    def clean_genres(x):
        if pd.isna(x): return ''
        x = str(x).strip()
        x = re.sub(r'[\[\]\'"()]', '', x)  # הסרת סוגריים וגרשיים משיירי ייצוא
        return x.strip().strip(',')
    if 'genres' in data.columns:
        data['genres_clean'] = data['genres'].apply(clean_genres)
    elif 'genres_clean' not in data.columns:
        data['genres_clean'] = ''
    # עמודה בינארית לכל ז'אנר עיקרי — רשימה קבועה מונעת חוסר התאמה בין Train לטסט
    GENRES_MAIN = [
        'Drama', 'Comedy', 'Romance', 'Action', 'Documentary',
        'Crime', 'Thriller', 'Horror', 'Adventure', 'Mystery',
        'Family', 'Fantasy', 'Biography', 'History', 'Sci-Fi',
        'Music', 'Musical', 'War', 'Animation', 'Sport',
        'Western', 'Adult', 'Film-Noir'
    ]
    GENRES_OTHER = ['News', 'Reality-TV', 'Talk-Show', 'Game-Show']
    for genre in GENRES_MAIN:
        data[f'genre_{genre}'] = data['genres_clean'].str.contains(
            genre, na=False).astype(int)
    # ז'אנרי טלוויזיה נישתיים — מאוגדים לעמודה אחת
    data['genre_other'] = data['genres_clean'].apply(
        lambda x: 1 if any(g in str(x) for g in GENRES_OTHER) else 0)
    # פיצ'ר נומרי: מספר הז'אנרים לסרט
    all_genre_cols = [f'genre_{g}' for g in GENRES_MAIN] + ['genre_other']
    data['genre_count'] = data[all_genre_cols].sum(axis=1)
    # ─── מספר שחקנים ─────────────────────────────────────────────────────────
    if 'lead_actors_ids' in data.columns:
        def count_actors(x):
            if not isinstance(x, str): return 0
            try:
                result = ast.literal_eval(x)  # פירוק בטוח של מחרוזת לרשימת Python
                return len(result) if isinstance(result, list) else 0
            except:
                return 0
        data['num_actors'] = data['lead_actors_ids'].apply(count_actors)
    elif 'num_actors' not in data.columns:
        data['num_actors'] = 0
    # ─── מדינה ושפה ───────────────────────────────────────────────────────────
    # קטגוריזציה ידנית — מבטיחה עמודות זהות בכל סט (בניגוד ל-get_dummies)
    def categorize_country(x):
        if pd.isna(x) or x == 'Not Found': return 'other'
        if x == 'United States': return 'usa'
        if x == 'India': return 'india'
        if x in ['United Kingdom', 'France', 'Italy', 'Canada', 'Spain']:
            return 'western'
        return 'other'

    def categorize_language(x):
        if pd.isna(x) or x == 'Not Found': return 'other'
        if x == 'English': return 'english'
        if x in ['French', 'Spanish', 'Italian', 'German']: return 'european'
        if x in ['Hindi', 'Tamil', 'Telugu']: return 'indian'
        return 'other'
    if 'Country' in data.columns:
        data['country_cat'] = data['Country'].apply(categorize_country)
    else:
        data['country_cat'] = 'other'
    if 'Language' in data.columns:
        data['language_cat'] = data['Language'].apply(categorize_language)
    else:
        data['language_cat'] = 'other'
    # המרה לעמודות בינאריות — ידנית ולא עם get_dummies
    for cat in ['india', 'usa', 'western']:
        data[f'country_{cat}'] = (data['country_cat'] == cat).astype(int)
    for cat in ['english', 'european', 'indian']:
        data[f'language_{cat}'] = (data['language_cat'] == cat).astype(int)
    # ─── תקציב ועלילה ─────────────────────────────────────────────────────────
    # פיצ'רים בינאריים בלבד — הערך המספרי חסר מדי לשימוש ישיר
    if 'budget' in data.columns:
        data['has_budget'] = data['budget'].notna().astype(int)
    elif 'has_budget' not in data.columns:
        data['has_budget'] = 0
    if 'plot' in data.columns:
        data['has_plot'] = (~data['plot'].isna()).astype(int)
    elif 'has_plot' not in data.columns:
        data['has_plot'] = 0
    # המרת bool ל-int — דרישת sklearn
    bool_cols = data.select_dtypes(include=['bool']).columns
    data[bool_cols] = data[bool_cols].astype(int)
    # ─── ניקוי עמודות ביניים ──────────────────────────────────────────────────
    # מחיקת כל העמודות הגולמיות שכבר עובדו לפיצ'רים
    drop_final = ['genres', 'genres_clean', 'lead_actors_ids',
                  'Country', 'Language', 'budget', 'plot',
                  'country_cat', 'language_cat',
                  'country_missing', 'language_missing',
                  'budget_clean', 'budget_cat']
    data = data.drop(columns=[c for c in drop_final if c in data.columns])
    # ─── יישור עמודות מול רשימה סטטית ────────────────────────────────────────
    # עמודות חסרות → מתמלאות ב-0; עמודות עודפות → נמחקות
    expected_cols = [
        'startYear', 'runtimeMinutes', 'genre_Drama', 'genre_Comedy',
        'genre_Romance', 'genre_Action', 'genre_Documentary', 'genre_Crime',
        'genre_Thriller', 'genre_Horror', 'genre_Adventure', 'genre_Mystery',
        'genre_Family', 'genre_Fantasy', 'genre_Biography', 'genre_History',
        'genre_Sci-Fi', 'genre_Music', 'genre_Musical', 'genre_War',
        'genre_Animation', 'genre_Sport', 'genre_Western', 'genre_Adult',
        'genre_Film-Noir', 'genre_other', 'genre_count', 'num_actors',
        'country_india', 'country_usa', 'country_western', 'language_english',
        'language_european', 'language_indian', 'has_budget', 'has_plot'
    ]
    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0

    data = data[expected_cols]
    return data
