from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap


def generate_random_indices(max_index, count):
    """
    Generate a list of random indices without replacement.

    Args:
        max_index: The maximum index value (exclusive)
        count: Number of indices to generate

    Returns:
        List of random indices
    """
    # Ensure we don't try to sample more indices than available
    count = min(count, max_index)
    return np.random.choice(max_index, size=count, replace=False).tolist()


def toggle_highlight(highlighted_indices):
    """
    Toggle highlight state and generate new random indices when activated.
    """
    if st.session_state.highlight_active:
        # Turn off highlighting
        st.session_state.highlighted_indices = []
        st.session_state.highlight_active = False
    else:
        # Turn on highlighting with new random indices
        # max_index = len(st.session_state.df)
        # # Generate around 10% of the data points, or at least 5
        # num_to_highlight = 5
        st.session_state.highlighted_indices = highlighted_indices
        st.session_state.highlight_active = True


def create_base_plot(df, scatter_args, marker_size, reduced_opacity=False):
    """
    Create and configure the base scatter plot.

    Args:
        df: DataFrame containing the data points
        scatter_args: Dictionary of arguments for px.scatter
        marker_size: Size of the markers
        reduced_opacity: Whether to reduce the opacity of points (for when highlighting is active)

    Returns:
        Configured plotly figure
    """
    fig = px.scatter(df, **scatter_args)

    # print(df.columns)

    # Customize the plot appearance
    opacity = (
        0.5 if reduced_opacity else 0.85
    )  # Reduce opacity when highlighting is active
    fig.update_traces(
        marker=dict(
            size=marker_size,
            opacity=opacity,
            line=dict(width=1),
        ),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        width=700,
        height=600,
    )

    return fig


def highlight_query_points(fig, df):
    """
    Add highlighted query points to the figure.

    Args:
        fig: Plotly figure to add traces to
        df: DataFrame containing the data
    """
    HIGHLIGHT_MARKER_SIZE = 20

    indices = df[df["_highlighted"] == True].index
    highlighted_df = df.iloc[indices]
    # Create a new trace for highlighted points with specified size
    for category in highlighted_df["source_name"].unique():
        category_df = highlighted_df[highlighted_df["source_name"] == category]
        if not category_df.empty:

            # Add the highlighted points
            fig.add_trace(
                px.scatter(
                    category_df,
                    x="x",
                    y="y",
                    color_discrete_sequence=[st.session_state.color_palette[category]],
                    hover_name="title",
                )
                .data[0]
                .update(
                    marker=dict(
                        size=HIGHLIGHT_MARKER_SIZE,
                        opacity=1.0,
                        line=dict(width=1, color="black"),
                    ),
                    showlegend=False,  # Don't add duplicate legend entries
                )
            )


def render_visualization_column():
    """
    Renders the left column with embedding visualization functionality.
    """
    st.title("Arquivo dos Cravos")

    # Define constants for marker sizing
    DEFAULT_MARKER_SIZE = 10

    # Get data and category information
    df = st.session_state.df
    categs = df["source_name"].unique().tolist()

    df["hover_text"] = df.apply(
        lambda row: f"Fonte: {row['source_name']}<br>"
        + f"{row['tstamp']}<br>"
        + f"Arquivo: {row['linkToArchive']}<br>"
        + f"URL: {row['linkToNoFrame']}",
        axis=1,
    )

    # print(df.columns)

    # Define scatter plot arguments
    scatter_args = dict(
        x="x",
        y="y",
        # color_discrete_sequence=px.colors.qualitative.Bold,
        color="source_name",
        # hover_name="title",
        category_orders={"category": categs},
        labels={"category": "Source"},
        color_discrete_map=st.session_state.color_palette,
        title="",
        hover_name="hover_text",
        custom_data=["hover_text"],
        hover_data={
            "hover_text": False,
            "x": False,  # Show x with 2 decimal places
            "y": False,  # Hide y from hover
            "source_name": False,  # Hide category from hover
        },
        # title="UMAP Projection of Document Embeddings",
    )

    controls_container = st.container()

    with controls_container:

        # UI controls for category selection
        categories = st.multiselect(
            "Select categories to display:",
            options=categs,
            default=categs,
        )

    # Filter dataframe based on selected categories
    filtered_df = df[df["source_name"].isin(categories)]
    col1, col2 = st.columns([2, 1])

    with col1:

        # Create the base visualization with reduced opacity when highlighting is active
        fig = create_base_plot(
            filtered_df,
            scatter_args,
            DEFAULT_MARKER_SIZE,
            reduced_opacity=st.session_state.highlight_active,
        )

        # Highlight query points if they exist

        # Add highlighted random points if active
        if st.session_state.highlight_active and st.session_state.highlighted_indices:
            highlight_query_points(fig, filtered_df)
            focus_on_highlights(fig, filtered_df, st.session_state.highlighted_indices)

        fig = apply_theme(fig)

        # Display the plot
        # st.plotly_chart(fig, use_container_width=True)

        selected_points = st.plotly_chart(
            fig, on_select="rerun", key="scatter_plot", use_container_width=True
        )

    # Render metadata
    with col2:
        # Create a container for the metadata
        metadata_container = st.container()

        with metadata_container:
            # print("st.session_state.metadata")
            # print(st.session_state.color_palette)
            if selected_points and selected_points["selection"]["points"]:
                # Display multiple points horizontally if multiple are selected
                num_points = len(selected_points["selection"]["points"])
                if num_points > 1:
                    cols = st.columns(min(num_points, 3))  # Max 3 columns

                for i, point in enumerate(selected_points["selection"]["points"]):
                    point_index = point["point_index"]
                    selected_row = df.iloc[point_index]

                    # If multiple points, use columns
                    if num_points > 1:
                        col_idx = i % 3
                        with cols[col_idx]:
                            display_metadata_card(selected_row, point_index)
                    else:
                        display_metadata_card(selected_row, point_index)
            else:
                st.info("Clique num ponto do gráfico para ver os seus detalhes aqui.")


def display_metadata_card(selected_row, point_index):
    res_id = selected_row["meta_id"]
    point_meta = {}
    meta_content = ""
    if st.session_state.metadata:
        for mi, metas in enumerate(st.session_state.metadata):
            # print(mi, metas)
            if metas["m_id"] == res_id:
                point_meta["link_arquivo"] = metas["link"]
                point_meta["content"] = st.session_state.documents[mi]

    meta_content = ".".join(point_meta["content"].split(".")[:2])

    color_palette = st.session_state.color_palette
    bgcolor = color_palette["BGColor2"]
    textcolor = color_palette["TextColor"]
    tstamp = str(selected_row["tstamp"])

    if selected_row["source_name"] in list(color_palette.keys()):
        color = color_palette[selected_row["source_name"]]
    else:
        color = "#444444"

    """Helper function to display metadata card"""
    st.markdown(
        f"""
    <div style="
        border: 8px solid {color}77;
        padding: 15px;
        margin: 10px 0;
        color: {textcolor};
        background-color: {color}44;
    ">
        <h4 style="color: {color}; margin-top: 0;">Detalhes</h4>
        <p><strong>Fonte:</strong> {selected_row['source_name']} ({tstamp[:4]}-{tstamp[4:6]}-{tstamp[6:8]})</p>
        <div>{meta_content} ...</div>
        <p><em><strong>Ler mais: </strong></em> <a href="{selected_row["linkToArchive"]}">{selected_row['linkToArchive']}</a></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def focus_on_highlights(fig, df, indices):
    """
    Adjust the plot's axes range to focus on the highlighted points.

    Args:
        fig (plotly.graph_objects.Figure): Figure to update
        df (pandas.DataFrame): DataFrame containing the data
        indices (list): List of indices for highlighted points
    """
    ZOOM_BUFFER_PERCENTAGE = 0.15

    if not indices:
        return

    # Get the subset of data for highlighted points
    highlighted_df = df.iloc[indices]

    if highlighted_df.empty:
        return

    # Calculate min/max x and y values for highlighted points
    min_x = highlighted_df["x"].min()
    max_x = highlighted_df["x"].max()
    min_y = highlighted_df["y"].min()
    max_y = highlighted_df["y"].max()

    # Calculate range to add buffer space
    x_range = max_x - min_x
    y_range = max_y - min_y

    # Add buffer to prevent zooming too tight
    buffer_x = max(x_range * ZOOM_BUFFER_PERCENTAGE, 0.5)
    buffer_y = max(y_range * ZOOM_BUFFER_PERCENTAGE, 0.5)

    # Calculate final axis ranges with buffer
    x_min = min_x - buffer_x
    x_max = max_x + buffer_x
    y_min = min_y - buffer_y
    y_max = max_y + buffer_y

    # Update figure's x and y axis ranges
    fig.update_layout(
        xaxis=dict(range=[x_min, x_max]), yaxis=dict(range=[y_min, y_max])
    )


def reset_zoom(fig):
    """
    Reset the plot's zoom level to show all data points.

    Args:
        fig (plotly.graph_objects.Figure): Figure to update
    """
    fig.update_layout(xaxis=dict(autorange=True), yaxis=dict(autorange=True))


def apply_theme(fig):
    is_dark_theme = st.session_state.dark

    # Get Streamlit theme colors
    primary_color = st.config.get_option("theme.primaryColor")
    background_color = st.config.get_option("theme.backgroundColor")
    secondary_bg_color = st.config.get_option("theme.secondaryBackgroundColor")
    text_color = st.config.get_option("theme.textColor")
    # Update plot theming

    fig.update_layout(
        paper_bgcolor=background_color,
        plot_bgcolor=secondary_bg_color,
        font_color=text_color,
        # Remove axis titles
        xaxis_title=None,
        yaxis_title=None,
        # Remove legend title
        legend_title_text=None,
        # # Remove plot background
        # plot_bgcolor='rgba(0,0,0,0)',
        # # Optional: Remove paper background (area around the plot)
        # paper_bgcolor='rgba(0,0,0,0)',
        # Remove grid
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        # Optional: Remove margins for a full bleed look
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.update_xaxes(
        gridcolor=text_color if is_dark_theme else "lightgrey",
        gridwidth=0.5 if is_dark_theme else 1,
    )
    fig.update_yaxes(
        gridcolor=text_color if is_dark_theme else "lightgrey",
        gridwidth=0.5 if is_dark_theme else 1,
    )

    return fig
