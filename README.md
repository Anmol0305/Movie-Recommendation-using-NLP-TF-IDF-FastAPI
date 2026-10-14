## 🎬 Try the Movie Recommendation System

🍿 **Explore Movie Recommendations:** [Launch Movie Recommender App](https://movie-recommendation-using-nlp-tf-idf-fastapi-jv4elshdlg3bxfz9.streamlit.app/)

**Technologies:** Python, NLP, TF-IDF, FastAPI, Streamlit 🎬

A movie discovery and recommendation app built with **Streamlit**, **FastAPI**, **Python**, **Pandas**, and **scikit-learn TF-IDF**. The app lets users search for movies, view movie details and discover similar titles. Movie metadata and posters are fetched through The Movie Database (TMDB) API.

## Project structure

```text
movie_recommendation_github_ready/
├── app.py                 # Streamlit frontend
├── main.py                # FastAPI backend
├── requirements.txt       # Python dependencies
├── .env.example           # Environment-variable template (no real key)
├── .gitignore
├── README.md
├── df.pkl                 # Movie dataframe used by the recommender
├── indices.pkl            # Movie title/index mapping
├── tfidf.pkl              # Fitted TF-IDF vectorizer
└── tfidf_matrix.pkl       # TF-IDF feature matrix
```

## Requirements

- Python 3.10–3.12
- A TMDB API key: https://www.themoviedb.org/settings/api

## Run locally (Windows / VS Code)

1. Open this project folder in VS Code.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and replace the placeholder with your real key:

   ```text
   TMDB_API_KEY=your_actual_tmdb_api_key
   ```

   **Never upload `.env` or your API key to GitHub.**

5. Start the backend in the first terminal:

   ```powershell
   uvicorn main:app --reload
   ```

   Check `http://127.0.0.1:8000/docs` to see the API documentation.

6. Open a second terminal, activate `.venv` there too, then start the frontend:

   ```powershell
   streamlit run app.py
   ```

7. Open the local Streamlit URL shown in the terminal (usually `http://localhost:8501`).

## Deploying

### FastAPI backend on Render

- Create a **Web Service** connected to this GitHub repository.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add an environment variable named `TMDB_API_KEY` in Render's Environment settings.
- Confirm the deployed backend responds at `/health` and `/docs`.

### Streamlit frontend on Streamlit Community Cloud

- Deploy `app.py` from the same GitHub repository.
- In `app.py`, change `API_BASE = "http://127.0.0.1:8000"` to your actual deployed Render backend base URL (for example, `https://your-service-name.onrender.com`). Do not add `/docs` to the base URL.
- Commit and push the change, then redeploy the Streamlit app.

## Important notes

- The four `.pkl` files are required by the backend. They are included in this repository package; keep them in the same folder as `main.py`.
- Only load pickle files that you created or trust.
- This repository does not include your TMDB API key, virtual environment, or Python cache files.
- The frontend is configured for local development by default. Public deployment requires setting `API_BASE` to your deployed backend URL.

## Technologies

Python · Streamlit · FastAPI · REST API · Pandas · NumPy · scikit-learn · TF-IDF · TMDB API


### GitHub browser upload note

The dataframe model is stored as `df.pkl.gz` to keep the individual file below GitHub browser upload limits. The FastAPI backend automatically decompresses it when loading the model. Keep `df.pkl.gz`, `indices.pkl`, `tfidf.pkl`, and `tfidf_matrix.pkl` together in the same folder as `main.py`.
