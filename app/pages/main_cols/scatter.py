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

    df = st.session_state.df
    categs = df["source_name"].unique().tolist()

    scatter_args = dict(
        x="x",
        y="y",
        color_discrete_sequence=px.colors.qualitative.Bold,
        color="source_name",
        hover_name="title",
        category_orders={"category": categs},
        labels={"category": "Category"},
        title="UMAP Projection of Document Embeddings",
    )

    categories = st.multiselect(
        "Select categories to display:",
        options=categs,
        default=categs,
    )

    # Create a Plotly scatter plot for the embeddings
    fig = px.scatter(st.session_state.df, **scatter_args)

    # Customize the plot appearance
    fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode="markers"))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        width=700,
        height=600,
    )

    # Highlight query point with different marker
    query_points = st.session_state.df[st.session_state.df["_highlighted"] == True]
    if not query_points.empty:
        fig.add_trace(
            px.scatter(
                query_points, x="x", y="y", color_discrete_sequence=["black"]
            ).data[0]
        )

    st.plotly_chart(fig, use_container_width=True)

    # Add options for filtering the visualization

    # if st.button("Apply Filter"):
    #     filtered_df = st.session_state.df[
    #         (st.session_state.df["source_name"].isin(categs))
    #         | (st.session_state.df["_highlighted"] == True)
    #     ]

    #     # filtered_df = st.session_state.df

    #     # Update the plot with filtered data
    #     fig = px.scatter(filtered_df, **scatter_args)

    #     fig.update_traces(
    #         marker=dict(size=10, opacity=0.7), selector=dict(mode="markers")
    #     )
    #     fig.update_layout(
    #         legend=dict(
    #             orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    #         ),
    #         plot_bgcolor="white",
    #         width=700,
    #         height=600,
    #     )

    #     # Highlight query point with different marker
    #     query_points = filtered_df[filtered_df["_highlighted"] == True]
    #     if not query_points.empty:
    #         fig.add_trace(
    #             px.scatter(
    #                 query_points, x="x", y="y", color_discrete_sequence=["black"]
    #             ).data[0]
    #         )

    #     st.plotly_chart(fig, use_container_width=True)
