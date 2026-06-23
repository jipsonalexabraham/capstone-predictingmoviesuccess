import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "movie_success_random_forest_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
DATA_PATH = ARTIFACT_DIR / "movie_success_dashboard_export.csv"

CLASS_ORDER = ["Flop", "Average", "Hit"]


st.set_page_config(page_title="Movie Success Predictor", layout="wide")
sns.set_theme(style="whitegrid", palette="Set2")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)
    return model, metadata


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def engineer_features(input_df):
    data = input_df.copy()
    data["movie_title"] = data.get("movie_title", pd.Series(["Untitled"])).astype(str).str.strip()
    data["genres"] = data.get("genres", pd.Series(["Unknown"])).fillna("Unknown")
    data["plot_keywords"] = data.get("plot_keywords", pd.Series([""])).fillna("")
    data["genre_count"] = data["genres"].apply(lambda value: len(str(value).split("|")))
    data["keyword_count"] = data["plot_keywords"].apply(lambda value: 0 if value == "" else len(str(value).split("|")))
    data["title_length"] = data["movie_title"].apply(len)
    data["main_genre"] = data["genres"].apply(lambda value: str(value).split("|")[0])
    data["profit"] = data["gross"] - data["budget"]
    data["roi"] = np.where(data["budget"] > 0, data["gross"] / data["budget"], np.nan)
    data["movie_age"] = 2026 - data["title_year"]
    return data


def money(value):
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def clean_text(value, default="Unknown"):
    if pd.isna(value) or str(value).strip() == "":
        return default
    return str(value)


def clean_float(value, default=0.0):
    if pd.isna(value):
        return float(default)
    return float(value)


def clean_int(value, default=0):
    if pd.isna(value):
        return int(default)
    return int(value)


def show_bar(data, x, y, title, color="#4c78a8"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=data, x=x, y=y, ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)


missing = [path for path in [MODEL_PATH, METADATA_PATH, DATA_PATH] if not path.exists()]
if missing:
    st.error("Model files are missing. Please run the notebook first.")
    st.write(missing)
    st.stop()


model, metadata = load_model()
df = load_data()

st.title("Movie Success Predictor")
st.write("This simple dashboard predicts whether a movie is likely to be a Hit, Average, or Flop based on movie details.")

dashboard_tab, predict_tab, data_tab = st.tabs(["Simple Dashboard", "Predict Movie", "View Data"])

with dashboard_tab:
    st.subheader("Project Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{len(df):,}")
    col2.metric("Average IMDB Score", f"{df['imdb_score'].mean():.2f}")
    col3.metric("Most Common Class", df["Classify"].mode()[0])
    col4.metric("Median Budget", money(df["budget"].median()))

    st.subheader("Movie Success Categories")
    class_count = df["Classify"].value_counts().reindex(CLASS_ORDER).fillna(0).reset_index()
    class_count.columns = ["Success Class", "Number of Movies"]
    show_bar(class_count, "Success Class", "Number of Movies", "How Many Movies Are Flop, Average, or Hit?")

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Top Movie Genres")
        genre_count = df["main_genre"].value_counts().head(8).reset_index()
        genre_count.columns = ["Genre", "Movies"]
        show_bar(genre_count, "Movies", "Genre", "Most Common Genres", "#59a14f")

    with col6:
        st.subheader("IMDB Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["imdb_score"], bins=20, kde=True, ax=ax, color="#f28e2b")
        ax.set_title("Distribution of IMDB Scores")
        ax.set_xlabel("IMDB Score")
        ax.set_ylabel("Number of Movies")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Best Rated Movies")
    top_movies = df.sort_values("imdb_score", ascending=False)[
        ["movie_title", "imdb_score", "Classify", "main_genre", "country", "title_year"]
    ].head(10)
    st.dataframe(top_movies, use_container_width=True, hide_index=True)

with predict_tab:
    st.subheader("Select a Movie")

    movie_names = sorted(df["movie_title"].dropna().astype(str).unique())
    selected_movie_name = st.selectbox("Movie Name", movie_names)
    selected_movie = df[df["movie_title"].astype(str) == selected_movie_name].iloc[0]

    st.write(
        f"Actual class: **{selected_movie['Classify']}** | "
        f"IMDB score: **{selected_movie['imdb_score']:.1f}** | "
        f"Genre: **{selected_movie['main_genre']}**"
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            movie_title = selected_movie_name
            genres = st.text_input("Genres", clean_text(selected_movie.get("genres"), "Unknown"))
            director_name = st.text_input("Director Name", clean_text(selected_movie.get("director_name"), "Unknown"))
            actor_1_name = st.text_input("Main Actor", clean_text(selected_movie.get("actor_1_name"), "Unknown"))
            country = st.text_input("Country", clean_text(selected_movie.get("country"), "USA"))
            language = st.text_input("Language", clean_text(selected_movie.get("language"), "English"))

        with col2:
            content_rating = st.text_input("Content Rating", clean_text(selected_movie.get("content_rating"), "Not Rated"))
            title_year = st.number_input("Release Year", min_value=1900, max_value=2026, value=clean_int(selected_movie.get("title_year"), 2026))
            duration = st.number_input("Duration in Minutes", min_value=1, value=clean_int(selected_movie.get("duration"), 120))
            budget = st.number_input("Budget", min_value=0.0, value=clean_float(selected_movie.get("budget"), 0), step=1_000_000.0)
            gross = st.number_input("Gross Revenue", min_value=0.0, value=clean_float(selected_movie.get("gross"), 0), step=1_000_000.0)
            votes = st.number_input("Number of Votes", min_value=0, value=clean_int(selected_movie.get("num_voted_users"), 0))

        submitted = st.form_submit_button("Predict")

    if submitted:
        movie = {
            "color": "Color",
            "director_name": director_name,
            "num_critic_for_reviews": selected_movie.get("num_critic_for_reviews", 100),
            "duration": duration,
            "director_facebook_likes": selected_movie.get("director_facebook_likes", 0),
            "actor_3_facebook_likes": selected_movie.get("actor_3_facebook_likes", 0),
            "actor_2_name": selected_movie.get("actor_2_name", "Unknown"),
            "actor_1_facebook_likes": selected_movie.get("actor_1_facebook_likes", 0),
            "gross": gross,
            "genres": genres,
            "actor_1_name": actor_1_name,
            "movie_title": movie_title,
            "num_voted_users": votes,
            "cast_total_facebook_likes": selected_movie.get("cast_total_facebook_likes", 0),
            "actor_3_name": selected_movie.get("actor_3_name", "Unknown"),
            "facenumber_in_poster": selected_movie.get("facenumber_in_poster", 0),
            "plot_keywords": selected_movie.get("plot_keywords", ""),
            "num_user_for_reviews": selected_movie.get("num_user_for_reviews", 0),
            "language": language,
            "country": country,
            "content_rating": content_rating,
            "budget": budget,
            "title_year": title_year,
            "actor_2_facebook_likes": selected_movie.get("actor_2_facebook_likes", 0),
            "aspect_ratio": selected_movie.get("aspect_ratio", 2.35),
            "movie_facebook_likes": selected_movie.get("movie_facebook_likes", 10_000),
        }

        input_data = engineer_features(pd.DataFrame([movie]))
        feature_columns = metadata["numeric_features"] + metadata["categorical_features"]
        input_data = input_data.reindex(columns=feature_columns)

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        probability_df = pd.DataFrame({"Class": model.named_steps["model"].classes_, "Probability": probabilities})

        st.success(f"Predicted Movie Category for {movie_title}: {prediction}")
        st.info(f"Actual Category in Dataset: {selected_movie['Classify']}")
        st.write("Prediction probability:")
        st.bar_chart(probability_df.set_index("Class"))

with data_tab:
    st.subheader("Movie Dataset")
    columns = ["movie_title", "imdb_score", "Classify", "predicted_class", "main_genre", "country", "title_year", "budget", "gross"]
    st.dataframe(df[columns].head(100), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Dashboard Data",
        df[columns].to_csv(index=False).encode("utf-8"),
        "movie_success_simple_dashboard.csv",
        "text/csv",
    )
