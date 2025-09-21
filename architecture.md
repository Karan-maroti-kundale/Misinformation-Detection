# Architecture (High-Level)

```
[User] ──> Streamlit App (app.py)
               │
               ├─ Text Path:
               │     ├─ Language detect & translate (utils.py)
               │     ├─ Preprocess (utils.basic_clean)
               │     ├─ Vectorizer + Logistic Regression (models/*)
               │     ├─ Explanation:
               │     │     ├─ local keywords (coef lookups)
               │     │     └─ (optional) Vertex AI summary
               │     └─ Literacy Tips (tips.py)
               │
               └─ Image Path:
                     ├─ Basic EXIF check (ExifRead)
                     ├─ Sharpness proxy (Laplacian variance)
                     ├─ pHash (imagehash)
                     ├─ Simple flags (small size, no EXIF, edited tags)
                     └─ Literacy Tips
```

**Why this design?**
- Fast to start, easy to extend.
- Transparent explanations (keywords + rules).
- Optional Google Cloud paths without breaking local dev.
