from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


ARTIFACT_DIR = Path("artifacts")
PLOT_DIR = ARTIFACT_DIR / "plots"
DASHBOARD_DATA = ARTIFACT_DIR / "movie_success_dashboard_export.csv"
TEST_PREDICTIONS = ARTIFACT_DIR / "test_predictions.csv"
CLASS_ORDER = ["Flop", "Average", "Hit"]
PALETTE = {"Flop": "#e15759", "Average": "#f28e2b", "Hit": "#59a14f"}


def save_plot(fig, filename):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def top_n_counts(data, column, n=10):
    return data[column].fillna("Unknown").astype(str).value_counts().head(n).index


def format_money_axis(ax):
    def money_label(value, _position):
        if abs(value) >= 1_000_000_000:
            return f"${value / 1_000_000_000:.1f}B"
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.0f}M"
        return f"${value:,.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(money_label))


def plot_success_breakdowns(df):
    dimensions = [
        ("country", "Top Countries", "01_success_by_country.png"),
        ("content_rating", "Content Ratings", "02_success_by_content_rating.png"),
        ("language", "Top Languages", "03_success_by_language.png"),
        ("main_genre", "Top Main Genres", "04_success_by_main_genre.png"),
    ]

    for column, title, filename in dimensions:
        subset = df[df[column].isin(top_n_counts(df, column))]
        counts = (
            subset.groupby([column, "predicted_class"])
            .size()
            .reset_index(name="movies")
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(
            data=counts,
            y=column,
            x="movies",
            hue="predicted_class",
            hue_order=CLASS_ORDER,
            palette=PALETTE,
            ax=ax,
        )
        ax.set_title(f"Predicted Success Count by {title}")
        ax.set_xlabel("Movies")
        ax.set_ylabel("")
        ax.legend(title="Predicted Class")
        save_plot(fig, filename)


def plot_financial_metrics(df):
    metrics = [
        ("budget", "Average Budget", "05_average_budget_by_predicted_class.png"),
        ("gross", "Average Gross Revenue", "06_average_gross_by_predicted_class.png"),
        ("profit", "Average Profit", "07_average_profit_by_predicted_class.png"),
        ("roi", "Average ROI", "08_average_roi_by_predicted_class.png"),
    ]
    summary = (
        df.groupby("predicted_class")[["budget", "gross", "profit", "roi"]]
        .mean()
        .reindex(CLASS_ORDER)
        .reset_index()
    )

    for metric, title, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=summary,
            x="predicted_class",
            y=metric,
            order=CLASS_ORDER,
            hue="predicted_class",
            hue_order=CLASS_ORDER,
            palette=PALETTE,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{title} by Predicted Class")
        ax.set_xlabel("")
        ax.set_ylabel("")
        if metric != "roi":
            format_money_axis(ax)
        save_plot(fig, filename)


def plot_top_hit_people(df):
    hits = df[df["predicted_class"] == "Hit"].copy()
    hits["director_name"] = hits["director_name"].fillna("").astype(str).str.strip()
    hits["actor_1_name"] = hits["actor_1_name"].fillna("").astype(str).str.strip()

    valid_directors = hits[~hits["director_name"].isin(["", "Unknown"])]
    valid_actors = hits[~hits["actor_1_name"].isin(["", "Unknown"])]

    directors = (
        valid_directors["director_name"]
        .value_counts()
        .head(12)
        .rename_axis("person")
        .reset_index(name="predicted_hits")
    )
    actors = (
        valid_actors["actor_1_name"]
        .value_counts()
        .head(12)
        .rename_axis("person")
        .reset_index(name="predicted_hits")
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=directors, x="predicted_hits", y="person", color="#4e79a7", ax=ax)
    ax.set_title("Top Directors by Predicted Hit Movies")
    ax.set_xlabel("Predicted Hits")
    ax.set_ylabel("")
    save_plot(fig, "09_top_directors_predicted_hits.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=actors, x="predicted_hits", y="person", color="#76b7b2", ax=ax)
    ax.set_title("Top Lead Actors by Predicted Hit Movies")
    ax.set_xlabel("Predicted Hits")
    ax.set_ylabel("")
    save_plot(fig, "10_top_actors_predicted_hits.png")


def plot_yearly_trend(df):
    trend = (
        df.dropna(subset=["title_year"])
        .assign(title_year=lambda data: data["title_year"].astype(int))
        .groupby(["title_year", "predicted_class"])
        .size()
        .reset_index(name="movies")
    )
    trend = trend[trend["title_year"] >= trend["title_year"].quantile(0.03)]

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.lineplot(
        data=trend,
        x="title_year",
        y="movies",
        hue="predicted_class",
        hue_order=CLASS_ORDER,
        palette=PALETTE,
        marker="o",
        ax=ax,
    )
    ax.set_title("Year-wise Trend of Predicted Movie Success Classes")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Movies")
    ax.legend(title="Predicted Class")
    save_plot(fig, "11_yearwise_predicted_success_trend.png")


def plot_actual_vs_predicted():
    predictions = pd.read_csv(TEST_PREDICTIONS)
    matrix = pd.crosstab(
        predictions["actual_class"],
        predictions["predicted_class"],
        rownames=["Actual Class"],
        colnames=["Predicted Class"],
    ).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Model Prediction Comparison: Actual vs Predicted")
    save_plot(fig, "12_actual_vs_predicted_confusion_matrix.png")


def main():
    sns.set_theme(style="whitegrid", palette="Set2")
    df = pd.read_csv(DASHBOARD_DATA)

    plot_success_breakdowns(df)
    plot_financial_metrics(df)
    plot_top_hit_people(df)
    plot_yearly_trend(df)
    plot_actual_vs_predicted()

    print(f"Saved required plots to {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()
