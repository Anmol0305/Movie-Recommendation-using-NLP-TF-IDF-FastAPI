import os
import pickle
import gzip
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# =========================
# ENV
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Add it to the .env file as "
        "TMDB_API_KEY=your_actual_api_key"
    )


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Suitable for local development; restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE FILE PATHS / GLOBALS
# =========================
DF_PATH = os.path.join(BASE_DIR, "df.pkl.gz")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# RESPONSE MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = Field(default_factory=list)


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILITIES
# =========================
def _norm_title(title: str) -> str:
    return str(title).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call TMDB and translate network/API failures into HTTP errors."""
    query_params = dict(params)
    query_params["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"{TMDB_BASE}{path}",
                params=query_params,
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(exc).__name__}: {exc}",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB error {response.status_code}: {response.text}",
        )

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="TMDB returned an invalid JSON response.",
        ) from exc


async def tmdb_cards_from_results(
    results: List[dict],
    limit: int = 20,
) -> List[TMDBMovieCard]:
    cards: List[TMDBMovieCard] = []

    for movie in (results or [])[:limit]:
        if not movie.get("id"):
            continue

        cards.append(
            TMDBMovieCard(
                tmdb_id=int(movie["id"]),
                title=movie.get("title") or movie.get("name") or "",
                poster_url=make_img_url(movie.get("poster_path")),
                release_date=movie.get("release_date"),
                vote_average=movie.get("vote_average"),
            )
        )

    return cards


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(
        f"/movie/{movie_id}",
        {"language": "en-US"},
    )

    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres") or [],
    )


async def tmdb_search_movies(
    query: str,
    page: int = 1,
) -> Dict[str, Any]:
    """Return TMDB's search response, including multiple results."""
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF HELPERS
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    Convert indices.pkl into a normalized title -> row-index dictionary.
    Supports a dictionary or a pandas Series-like object with .items().
    """
    if not hasattr(indices, "items"):
        raise RuntimeError(
            "indices.pkl must be a dictionary or pandas Series-like object."
        )

    title_to_idx: Dict[str, int] = {}

    for title, index in indices.items():
        title_to_idx[_norm_title(title)] = int(index)

    return title_to_idx


def get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF index map is not initialized.",
        )

    normalized_title = _norm_title(title)

    if normalized_title in TITLE_TO_IDX:
        return TITLE_TO_IDX[normalized_title]

    raise HTTPException(
        status_code=404,
        detail=f"Title not found in local dataset: '{title}'",
    )


def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Return (title, score) recommendations using the precomputed TF-IDF matrix.
    This assumes the matrix row order matches the indices and df row order.
    """
    if df is None or tfidf_matrix is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF resources are not loaded.",
        )

    idx = get_local_idx_by_title(query_title)

    if idx < 0 or idx >= tfidf_matrix.shape[0] or idx >= len(df):
        raise HTTPException(
            status_code=500,
            detail="The saved movie index does not match the loaded data.",
        )

    query_vector = tfidf_matrix[idx]
    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
    order = np.argsort(-scores)

    recommendations: List[Tuple[str, float]] = []

    for row_index in order:
        row_index = int(row_index)

        if row_index == idx:
            continue

        try:
            movie_title = str(df.iloc[row_index]["title"])
        except (IndexError, KeyError, TypeError):
            continue

        recommendations.append((movie_title, float(scores[row_index])))

        if len(recommendations) >= top_n:
            break

    return recommendations


async def attach_tmdb_card_by_title(
    title: str,
) -> Optional[TMDBMovieCard]:
    """Look up a local recommendation on TMDB to retrieve poster metadata."""
    try:
        movie = await tmdb_search_first(title)

        if not movie:
            return None

        return TMDBMovieCard(
            tmdb_id=int(movie["id"]),
            title=movie.get("title") or title,
            poster_url=make_img_url(movie.get("poster_path")),
            release_date=movie.get("release_date"),
            vote_average=movie.get("vote_average"),
        )
    except (HTTPException, KeyError, TypeError, ValueError):
        # The local recommendation can still be returned without TMDB metadata.
        return None


# =========================
# STARTUP: LOAD PICKLE FILES
# =========================
@app.on_event("startup")
def load_pickles() -> None:
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    required_files = {
        "df.pkl.gz": DF_PATH,
        "indices.pkl": INDICES_PATH,
        "tfidf_matrix.pkl": TFIDF_MATRIX_PATH,
        "tfidf.pkl": TFIDF_PATH,
    }

    for filename, file_path in required_files.items():
        if not os.path.isfile(file_path):
            raise RuntimeError(
                f"Required file '{filename}' was not found in: {BASE_DIR}"
            )

    # Only load pickle files that you created or otherwise trust.
    with gzip.open(DF_PATH, "rb") as file:
        df = pickle.load(file)

    with open(INDICES_PATH, "rb") as file:
        indices_obj = pickle.load(file)

    with open(TFIDF_MATRIX_PATH, "rb") as file:
        tfidf_matrix = pickle.load(file)

    with open(TFIDF_PATH, "rb") as file:
        tfidf_obj = pickle.load(file)

    if not isinstance(df, pd.DataFrame) or "title" not in df.columns:
        raise RuntimeError(
            "df.pkl.gz must contain a pandas DataFrame with a 'title' column."
        )

    if not hasattr(tfidf_matrix, "shape") or tfidf_matrix.shape[0] != len(df):
        raise RuntimeError(
            "The TF-IDF matrix row count must match the number of rows in df.pkl."
        )

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)


# =========================
# ROUTES
# =========================
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
) -> List[TMDBMovieCard]:
    """
    Home feed categories:
    trending, popular, top_rated, upcoming, now_playing.
    """
    try:
        if category == "trending":
            data = await tmdb_get(
                "/trending/movie/day",
                {"language": "en-US"},
            )
        else:
            allowed_categories = {
                "popular",
                "top_rated",
                "upcoming",
                "now_playing",
            }

            if category not in allowed_categories:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid category.",
                )

            data = await tmdb_get(
                f"/movie/{category}",
                {"language": "en-US", "page": 1},
            )

        return await tmdb_cards_from_results(
            data.get("results", []),
            limit=limit,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Home route failed: {exc}",
        ) from exc


# ---------- TMDB KEYWORD SEARCH ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
) -> Dict[str, Any]:
    """Return multiple TMDB search results."""
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS ----------
@app.get(
    "/movie/id/{tmdb_id}",
    response_model=TMDBMovieDetails,
)
async def movie_details_route(tmdb_id: int) -> TMDBMovieDetails:
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get(
    "/recommend/genre",
    response_model=List[TMDBMovieCard],
)
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
) -> List[TMDBMovieCard]:
    """Recommend popular movies from the selected movie's first genre."""
    details = await tmdb_movie_details(tmdb_id)

    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )

    cards = await tmdb_cards_from_results(
        discover.get("results", []),
        limit=limit,
    )
    return [card for card in cards if card.tmdb_id != tmdb_id]


# ---------- TF-IDF RECOMMENDATIONS ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
) -> List[Dict[str, Any]]:
    recommendations = tfidf_recommend_titles(title, top_n=top_n)
    return [
        {"title": movie_title, "score": score}
        for movie_title, score in recommendations
    ]


# ---------- BUNDLE: DETAILS + TF-IDF + GENRE RECOMMENDATIONS ----------
@app.get(
    "/movie/search",
    response_model=SearchBundleResponse,
)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
) -> SearchBundleResponse:
    """
    Return movie details, local TF-IDF recommendations with TMDB metadata,
    and popular movies from the selected movie's first genre.
    For multiple search matches, use /tmdb/search.
    """
    best_match = await tmdb_search_first(query)

    if not best_match:
        raise HTTPException(
            status_code=404,
            detail=f"No TMDB movie found for query: {query}",
        )

    tmdb_id = int(best_match["id"])
    details = await tmdb_movie_details(tmdb_id)

    # TF-IDF recommendations: try the TMDB title, then the original query.
    tfidf_items: List[TFIDFRecItem] = []
    recommendations: List[Tuple[str, float]] = []

    try:
        recommendations = tfidf_recommend_titles(
            details.title,
            top_n=tfidf_top_n,
        )
    except HTTPException:
        try:
            recommendations = tfidf_recommend_titles(
                query,
                top_n=tfidf_top_n,
            )
        except HTTPException:
            recommendations = []

    for movie_title, score in recommendations:
        tmdb_card = await attach_tmdb_card_by_title(movie_title)
        tfidf_items.append(
            TFIDFRecItem(
                title=movie_title,
                score=score,
                tmdb=tmdb_card,
            )
        )

    # Genre recommendations from TMDB.
    genre_recommendations: List[TMDBMovieCard] = []

    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )

        cards = await tmdb_cards_from_results(
            discover.get("results", []),
            limit=genre_limit,
        )
        genre_recommendations = [
            card for card in cards if card.tmdb_id != details.tmdb_id
        ]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recommendations,
    )
