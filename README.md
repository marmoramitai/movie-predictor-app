# 🎬 IMDb Rating Predictor — Flask App

אפליקציית ווב לחיזוי דירוג IMDb של סרטים בעזרת מודל Elastic Net שאומן על 115,000+ סרטים.

\---

## 📁 מבנה הפרויקט

```
movie-predictor-app/
├── api.py                  # שרת Flask (endpoints: GET / ו-POST /predict)
├── assets\\\\\\\\\\\\\\\_data\\\\\\\\\\\\\\\_prep.py     # פונקציית prepare\\\\\\\\\\\\\\\_data — עיבוד נתונים ל-36 פיצ'רים
├── trained\\\\\\\\\\\\\\\_model.pkl       # Pipeline מאומן: SimpleImputer → StandardScaler → ElasticNet
├── requirements.txt        # תלויות Python
├── README.md               # קובץ זה
└── templates/
    └── index.html          # ממשק משתמש (Fetch API, ללא רענון דף)
```

\---

## ⚙️ הפעלה

### 1\. התקנת סביבה וירטואלית (מומלץ)

```bash
python -m venv venv
# Windows:
venv\\\\\\\\\\\\\\\\Scripts\\\\\\\\\\\\\\\\activate
# macOS / Linux:
source venv/bin/activate
```

### 2\. התקנת תלויות

```bash
pip install -r requirements.txt
```

### 3\. הפעלת השרת

```bash
python api.py
```

### 4\. פתיחת הדפדפן

```
http://localhost:5000
```

\---

## 📋 שדות הטופס

|שדה|חובה|תיאור|דוגמה|
|-|-|-|-|
|`startYear`|✅|שנת יציאת הסרט|`2015`|
|`runtimeMinutes`|❌|אורך הסרט בדקות|`120`|
|`genres`|❌|ז'אנרים מופרדים בפסיקים|`Action, Drama`|
|`Country`|❌|מדינת הפקה|`United States`|
|`Language`|❌|שפת הסרט|`English`|
|`lead\\\\\\\\\\\\\\\_actors\\\\\\\\\\\\\\\_ids`|❌|מזהי IMDb של שחקנים ראשיים|`\\\\\\\\\\\\\\\['nm0000148', 'nm0000204']`|
|`budget`|❌|תקציב בדולרים|`50000000`|
|`plot`|❌|תקציר קצר של העלילה|`A story about...`|

> שדות ריקים מטופלים אוטומטית (NaN / ברירת מחדל) ולא גורמים לשגיאה.

\---

## 🤖 המודל

|פרמטר|ערך|
|-|-|
|אלגוריתם|Elastic Net|
|alpha|0.01|
|l1\_ratio|0.1|
|Pipeline|SimpleImputer (median) → StandardScaler → ElasticNet|
|פיצ'רים|36|
|R² (CV)|0.2389 ± 0.0046|
|MAE (CV)|0.8643 ± 0.0097|
|RMSE (CV)|1.1279 ± 0.0120|
|נתוני אימון|115,560 סרטים|

### 36 הפיצ'רים (בסדר):

`startYear`, `runtimeMinutes`, `genre\\\\\\\\\\\\\\\_Drama`, `genre\\\\\\\\\\\\\\\_Comedy`, `genre\\\\\\\\\\\\\\\_Romance`, `genre\\\\\\\\\\\\\\\_Action`, `genre\\\\\\\\\\\\\\\_Documentary`, `genre\\\\\\\\\\\\\\\_Crime`, `genre\\\\\\\\\\\\\\\_Thriller`, `genre\\\\\\\\\\\\\\\_Horror`, `genre\\\\\\\\\\\\\\\_Adventure`, `genre\\\\\\\\\\\\\\\_Mystery`, `genre\\\\\\\\\\\\\\\_Family`, `genre\\\\\\\\\\\\\\\_Fantasy`, `genre\\\\\\\\\\\\\\\_Biography`, `genre\\\\\\\\\\\\\\\_History`, `genre\\\\\\\\\\\\\\\_Sci-Fi`, `genre\\\\\\\\\\\\\\\_Music`, `genre\\\\\\\\\\\\\\\_Musical`, `genre\\\\\\\\\\\\\\\_War`, `genre\\\\\\\\\\\\\\\_Animation`, `genre\\\\\\\\\\\\\\\_Sport`, `genre\\\\\\\\\\\\\\\_Western`, `genre\\\\\\\\\\\\\\\_Adult`, `genre\\\\\\\\\\\\\\\_Film-Noir`, `genre\\\\\\\\\\\\\\\_other`, `genre\\\\\\\\\\\\\\\_count`, `num\\\\\\\\\\\\\\\_actors`, `country\\\\\\\\\\\\\\\_india`, `country\\\\\\\\\\\\\\\_usa`, `country\\\\\\\\\\\\\\\_western`, `language\\\\\\\\\\\\\\\_english`, `language\\\\\\\\\\\\\\\_european`, `language\\\\\\\\\\\\\\\_indian`, `has\\\\\\\\\\\\\\\_budget`, `has\\\\\\\\\\\\\\\_plot`

\---

## 🔌 API

### `GET /`

מחזיר את דף ה-HTML.

### `POST /predict`

**Request body (JSON):**

```json
{
  "startYear": "2015",
  "runtimeMinutes": "120",
  "genres": "Action, Drama",
  "Country": "United States",
  "Language": "English",
  "lead\\\\\\\\\\\\\\\_actors\\\\\\\\\\\\\\\_ids": "\\\\\\\\\\\\\\\['nm0000148']",
  "budget": "50000000",
  "plot": "A thrilling story about..."
}
```

**Response (200):**

```json
{ "prediction": 7.23 }
```

**Response (400) — שגיאת קלט:**

```json
{ "error": "שגיאת קלט", "fields": { "startYear": "חייב להיות מספר שנה." } }
```

**Response (500) — שגיאת שרת:**

```json
{ "error": "שגיאה בחיזוי: ..." }
```

\---

## 👥 חברי הצוות

אמיתי מרמור

\---

## 📌 הערות

* הדירוג המחוזה מוצמד לטווח **1.0 – 10.0** (טווח IMDb החוקי).
* אם `startYear` קטן מ-1900, הוא מוחלף ב-NaN והמודל מטפל בו דרך ה-Imputer.
* האפליקציה **לא שומרת** נתונים — כל בקשה עצמאית.

