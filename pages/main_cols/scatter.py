from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap


def render_visualization_column():
    """
    Renders the left column with embedding visualization functionality.
    """
    st.title("Embedding Visualization")

    # Create a Plotly scatter plot for the embeddings
    fig = px.scatter(
        st.session_state.df,
        x="x",
        y="y",
        color="category",
        hover_name="source",
        color_discrete_sequence=px.colors.qualitative.Bold,
        category_orders={
            "category": ["Technology", "Science", "Arts", "History", "Current Query"]
        },
        labels={"category": "Category"},
        title="UMAP Projection of Document Embeddings",
    )

    # Customize the plot appearance
    fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode="markers"))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        width=700,
        height=600,
    )

    # Highlight query point with different marker
    query_points = st.session_state.df[
        st.session_state.df["category"] == "Current Query"
    ]
    if not query_points.empty:
        fig.add_trace(
            px.scatter(
                query_points, x="x", y="y", color_discrete_sequence=["black"]
            ).data[0]
        )

    st.plotly_chart(fig, use_container_width=True)

    # Add options for filtering the visualization
    # st.subheader("Filter Visualization")
    # categories = st.multiselect(
    #     "Select categories to display:",
    #     options=["Technology", "Science", "Arts", "History", "Current Query"],
    #     default=["Technology", "Science", "Arts", "History", "Current Query"],
    # )

    if st.button("Apply Filter"):
        # filtered_df = st.session_state.df[
        #     st.session_state.df["category"].isin(categories)
        # ]

        filtered_df = st.session_state.df

        # Update the plot with filtered data
        fig = px.scatter(
            filtered_df,
            x="x",
            y="y",
            color="category",
            hover_name="source",
            color_discrete_sequence=px.colors.qualitative.Bold,
            category_orders={
                "category": [
                    "Technology",
                    "Science",
                    "Arts",
                    "History",
                    "Current Query",
                ]
            },
            labels={"category": "Category"},
            title="Filtered UMAP Projection of Document Embeddings",
        )

        fig.update_traces(
            marker=dict(size=10, opacity=0.7), selector=dict(mode="markers")
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            plot_bgcolor="white",
            width=700,
            height=600,
        )

        # Highlight query point with different marker
        query_points = filtered_df[filtered_df["category"] == "Current Query"]
        if not query_points.empty:
            fig.add_trace(
                px.scatter(
                    query_points, x="x", y="y", color_discrete_sequence=["black"]
                ).data[0]
            )

        st.plotly_chart(fig, use_container_width=True)


# def create_embeddings(documents: List[Dict]) -> np.ndarray:
#     """Create embeddings for the document content."""
#     # Using a lightweight model for demo purposes
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     texts = [doc["content"] for doc in documents]
#     embeddings = model.encode(texts)
#     return embeddings


def project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D using UMAP."""
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )
    return reducer.fit_transform(embeddings)


def create_dataframe(documents: List[Dict], projections: np.ndarray) -> pd.DataFrame:
    """Create a dataframe with document information and projections."""
    df = pd.DataFrame(documents)
    df["x"] = projections[:, 0]
    df["y"] = projections[:, 1]
    return df
