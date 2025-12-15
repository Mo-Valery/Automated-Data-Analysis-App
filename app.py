import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io, base64

# ----------------------------------
# CONFIG
# ----------------------------------
st.set_page_config(page_title="Data Analysis App", layout="wide")

# ----------------------------------
# FUNCTIONS
# ----------------------------------
def convert_to_html(fig):
    """Convert a Matplotlib figure to HTML-embedded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}'/>"

def convert_plotly_fig_to_html(fig):
    """Convert a Plotly figure to HTML snippet."""
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def generate_html_report(df, plots_html, insights_html):
    """Generate full HTML report."""
    return f"""
    <html>
    <head>
        <title>Dataset Report</title>
        <style>
            body {{ font-family: Arial; padding: 20px; }}
            h1, h2 {{ color: #333; }}
        </style>
    </head>
    <body>
    <h1>Dataset Analysis Report</h1>
    <h2>Dataset Overview</h2>
    {df.head().to_html()}
    <h2>Visualisations</h2>
    {plots_html}
    <h2>Automatic Insights</h2>
    {insights_html}
    </body>
    </html>
    """

def generate_insights(df):
    """Generate automatic insights including visualization interpretations."""
    insights = ""
    numeric_df = df.select_dtypes(include=["number"])
    cat_df = df.select_dtypes(include=["object"])
    date_cols = df.select_dtypes(include=["datetime"])

    # Numeric summaries
    if not numeric_df.empty:
        insights += "<h3>Numeric Data Summary:</h3><ul>"
        for col in numeric_df.columns:
            mean = numeric_df[col].mean()
            median = numeric_df[col].median()
            std = numeric_df[col].std()
            min_val = numeric_df[col].min()
            max_val = numeric_df[col].max()
            insights += f"<li><b>{col}</b>: mean={mean:.2f}, median={median:.2f}, std={std:.2f}, min={min_val}, max={max_val}</li>"
        insights += "</ul>"

        # Correlation insights
        if numeric_df.shape[1] >= 2:
            corr_matrix = numeric_df.corr()
            high_corr = corr_matrix.abs().stack().reset_index()
            high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
            high_corr = high_corr.sort_values(0, ascending=False).head(5)
            insights += "<h3>Top Correlated Numeric Pairs:</h3><ul>"
            for _, row in high_corr.iterrows():
                insights += f"<li>{row['level_0']} & {row['level_1']}: correlation={row[0]:.2f} (strong relationship)</li>"
            insights += "</ul>"

        # Histogram insights for first numeric column
        first_col = numeric_df.columns[0]
        skew = numeric_df[first_col].skew()
        insights += f"<p>The distribution of <b>{first_col}</b> shows a skewness of {skew:.2f}, indicating {'right' if skew>0 else 'left'} skew.</p>"

    # Categorical summaries
    if not cat_df.empty:
        insights += "<h3>Categorical Data Summary:</h3><ul>"
        for col in cat_df.columns:
            top_val = cat_df[col].value_counts().idxmax()
            top_count = cat_df[col].value_counts().max()
            prop = top_count / len(df) * 100
            insights += f"<li><b>{col}</b>: most frequent='{top_val}' ({top_count} occurrences, {prop:.1f}% of total)</li>"
        insights += "</ul>"

    # Date trends
    if len(date_cols.columns) > 0 and not numeric_df.empty:
        date_col = date_cols.columns[0]
        value_col = numeric_df.columns[0]

        # Weekly trend
        df["week"] = df[date_col].dt.to_period("W").apply(lambda r: r.start_time)
        wk = df.groupby("week")[value_col].mean()
        max_week = wk.idxmax().strftime("%Y-%m-%d")
        min_week = wk.idxmin().strftime("%Y-%m-%d")
        insights += f"<p>The weekly trend of <b>{value_col}</b> peaks in the week starting {max_week} and is lowest in the week starting {min_week}.</p>"

        # Monthly trend
        df["month"] = df[date_col].dt.to_period("M").astype(str)
        mon = df.groupby("month")[value_col].mean()
        max_mon = mon.idxmax()
        min_mon = mon.idxmin()
        insights += f"<p>The monthly trend of <b>{value_col}</b> peaks in {max_mon} and is lowest in {min_mon}.</p>"

    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        insights += f"<p>There are {missing} missing values in the dataset.</p>"

    return insights or "<p>No significant insights found.</p>"

# ----------------------------------
# UI
# ----------------------------------
st.title("📊 Automated Data Analysis App")

uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

if uploaded:
    # Load dataset
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.success("File uploaded successfully!")

    # Standardize columns
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    # Convert numeric-like strings
    df = df.apply(pd.to_numeric, errors="ignore")

    # Convert date columns
    for col in df.columns:
        if "date" in col or "time" in col:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    # Dataset preview
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Interactive Data Cleaning
    # -----------------------------
    st.subheader("🧹 Data Cleaning")
    if st.checkbox("Clean dataset automatically?"):
        # Remove duplicates
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - df.shape[0]
        st.write(f"✅ Removed {duplicates_removed} duplicate rows.")

        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                st.write(f"✅ Filled missing values in numeric column '{col}' with median.")

        # Fill categorical missing values with mode
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                st.write(f"✅ Filled missing values in categorical column '{col}' with mode.")

        st.success("Data cleaning completed!")

    # Show cleaned dataset
    st.subheader("📝 Cleaned Dataset Preview")
    st.dataframe(df.head())

    # HTML collector
    plots_html = ""

    # -----------------------------------------
    # 1. Missing Values
    # -----------------------------------------
    st.subheader("📌 Missing Values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        missing[missing > 0].plot(kind="bar", ax=ax)
        ax.set_title("Missing Values per Column")
        st.pyplot(fig)
        plots_html += convert_to_html(fig)
    else:
        st.info("No missing values found.")

    # -----------------------------------------
    # 2. Correlation Heatmap
    # -----------------------------------------
    st.subheader("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric_df.corr(), cmap="viridis", annot=True, ax=ax)
        st.pyplot(fig)
        plots_html += convert_to_html(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    # -----------------------------------------
    # 3. Pairplot
    # -----------------------------------------
    st.subheader("📌 Pairplot (Top 3 Numeric Columns)")
    if numeric_df.shape[1] >= 3:
        selected = numeric_df.columns[:3]
        fig = sns.pairplot(df[selected])
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf)
        encoded = base64.b64encode(buf.getvalue()).decode()
        plots_html += f"<img src='data:image/png;base64,{encoded}'/>"
    else:
        st.info("Not enough numeric columns for pairplot.")

    # -----------------------------------------
    # 4. Histogram
    # -----------------------------------------
    st.subheader("📊 Distribution of First Numeric Column")
    if numeric_df.shape[1] > 0:
        colname = numeric_df.columns[0]
        fig = px.histogram(df, x=colname, nbins=40, title=f"Distribution of {colname}")
        st.plotly_chart(fig)
        plots_html += convert_plotly_fig_to_html(fig)

    # -----------------------------------------
    # 5. Weekly Trend
    # -----------------------------------------
    date_cols = df.select_dtypes(include=["datetime"])
    if len(date_cols.columns) > 0 and len(numeric_df.columns) > 0:
        date_col = date_cols.columns[0]
        value_col = numeric_df.columns[0]

        df["week"] = df[date_col].dt.to_period("W").apply(lambda r: r.start_time)
        wk = df.groupby("week")[value_col].mean().reset_index()

        st.subheader("📅 Weekly Trend")
        fig = px.line(wk, x="week", y=value_col, title=f"Weekly Trend of {value_col}")
        st.plotly_chart(fig)
        plots_html += fig.to_html(full_html=False)
    else:
        st.info("No date column found for weekly trend.")

    # -----------------------------------------
    # 6. Monthly Trend
    # -----------------------------------------
    if len(date_cols.columns) > 0 and len(numeric_df.columns) > 0:
        df["month"] = df[date_col].dt.to_period("M").astype(str)
        mon = df.groupby("month")[value_col].mean().reset_index()

        st.subheader("📆 Monthly Trend")
        fig = px.bar(mon, x="month", y=value_col, title=f"Monthly Trend of {value_col}")
        st.plotly_chart(fig)
        plots_html += fig.to_html(full_html=False)

    # -----------------------------------------
    # 7. Categorical Counts
    # -----------------------------------------
    st.subheader("🏷 Top Categorical Values")
    cat_cols = df.select_dtypes(include=["object"])
    if len(cat_cols.columns) > 0:
        colname = cat_cols.columns[0]
        topcats = df[colname].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(7, 4))
        topcats.plot(kind="bar", ax=ax)
        ax.set_title(f"Top Categories in {colname}")
        st.pyplot(fig)
        plots_html += convert_to_html(fig)

    # -----------------------------------------
    # 8. Automatic Insights
    # -----------------------------------------
    st.subheader("💡 Automatic Insights")
    insights_html = generate_insights(df)
    st.markdown(insights_html, unsafe_allow_html=True)

    # -----------------------------------------
    # DOWNLOAD REPORT
    # -----------------------------------------
    st.subheader("📥 Download HTML Report")
    html_content = generate_html_report(df, plots_html, insights_html)
    st.download_button(
        "Download Report",
        data=html_content,
        file_name="analysis_report.html",
        mime="text/html"
    )
