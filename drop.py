import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np

st.set_page_config(page_title="📉 Traffic Drop Analyzer", layout="wide")
st.title("📉 Traffic Drop Analyzer")

# --- Instructions for Users ---
st.markdown("""
## Instructions
1. **Upload a File**: Please upload a CSV or Excel file containing your traffic data.
2. **Required Columns**:
   - The first column should contain page names or identifiers.
   - Subsequent columns should contain traffic data for different dates or months.
3. **Data Format**: Ensure that the traffic data columns contain numeric values.

### Example Format
| Page Name | Jan 2023 | Feb 2023 | Mar 2023 |
|-----------|----------|----------|----------|
| Page A    | 100      | 150      | 120      |
| Page B    | 200      | 180      | 160      |

### How to Use
- Use the sidebar to select the date range and set thresholds for traffic drop and gain.
- View the analysis results in the main area, including traffic trends and alerts.
- Download reports and charts using the provided buttons.
""")

# --- Safe column name cleaner ---
def deduplicate_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        col_str = str(col).strip()
        if col_str in seen:
            seen[col_str] += 1
            new_cols.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            new_cols.append(col_str)
    return new_cols

# --- Sparkline helper ---
def sparkline(data, figsize=(2, 0.25)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data, color='darkorange')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

# --- Upload section ---
uploaded_file = st.file_uploader("Upload Google Analytics / GSC file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read uploaded file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = deduplicate_columns(df.columns)

        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df.head(10))

        page_col = df.columns[0]
        date_cols = df.columns[1:]
        df[date_cols] = df[date_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        st.sidebar.header("📅 Date Range Selection")
        all_dates = list(date_cols)

        if len(all_dates) < 2:
            st.error("❌ At least two date columns are required.")
        else:
            # Set safe default indices
            default_prev_idx = max(0, len(all_dates) - 2)
            default_latest_idx = len(all_dates) - 1

            prev_month = st.sidebar.selectbox("Select Previous Month", all_dates[:-1], index=default_prev_idx)
            remaining_dates = all_dates[all_dates.index(prev_month)+1:]
            if not remaining_dates:
                st.error("❌ No valid month available after selected previous month.")
            else:
                latest_month = st.sidebar.selectbox("Select Latest Month", remaining_dates, index=0)

                start_idx = all_dates.index(prev_month)
                end_idx = all_dates.index(latest_month)
                trend_cols = all_dates[start_idx:end_idx+1]

                min_traffic = st.sidebar.number_input("Minimum traffic threshold", min_value=0, value=50)
                min_drop = st.sidebar.slider("Minimum drop % to show", 0, 100, 10, step=1)

                df["Drop %"] = np.where(
                    df[prev_month] > 0,
                    ((df[prev_month] - df[latest_month]) / df[prev_month]) * 100,
                    0
                ).round(2)

                df["Gain %"] = np.where(
                    df[prev_month] > 0,
                    ((df[latest_month] - df[prev_month]) / df[prev_month]) * 100,
                    0
                ).round(2)

                filtered_df = df[df[prev_month] >= min_traffic]
                drop_df = filtered_df[filtered_df["Drop %"] >= min_drop].sort_values("Drop %", ascending=False)
                gain_df = filtered_df[filtered_df["Gain %"] >= min_drop].sort_values("Gain %", ascending=False)

                st.sidebar.header("📊 Summary")
                st.sidebar.markdown(f"- Total pages analyzed: **{len(df)}**")
                st.sidebar.markdown(f"- Pages with ≥ {min_drop}% drop: **{len(drop_df)}**")
                st.sidebar.markdown(f"- Pages with ≥ {min_drop}% gain: **{len(gain_df)}**")

                st.subheader(f"📉 Pages with ≥ {min_drop}% Traffic Drop (≥ {min_traffic} visits in {prev_month})")

                if not drop_df.empty:
                    for idx, row in drop_df.iterrows():
                        st.markdown(f"**{row[page_col]}**")
                        plt_fig = plt.figure(figsize=(6, 2))
                        plt.plot(trend_cols, row[trend_cols].values, marker='o', color='darkorange')
                        plt.title(f"Trend: {row[page_col]}")
                        plt.grid(True)
                        st.pyplot(plt_fig)

                    csv = drop_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Drop Report CSV", csv, "traffic_drop_report.csv", "text/csv")
                else:
                    st.info("No pages match the current drop and traffic filters.")

                st.subheader(f"📈 Pages with ≥ {min_drop}% Traffic Gain (≥ {min_traffic} visits in {prev_month})")

                if not gain_df.empty:
                    st.dataframe(gain_df[[page_col, prev_month, latest_month, "Gain %"]])
                else:
                    st.info("No pages with significant traffic gains found.")

                st.subheader("📊 Page-wise Traffic Trend Comparison")
                pages_for_trend = st.multiselect("Select one or more pages to compare", df[page_col].tolist())

                if pages_for_trend:
                    plt.figure(figsize=(12, 5))
                    for page in pages_for_trend:
                        row = df[df[page_col] == page].iloc[0]
                        plt.plot(trend_cols, row[trend_cols].values, marker='o', label=page)
                    plt.title("Traffic Trends Comparison")
                    plt.xlabel("Month")
                    plt.ylabel("Traffic")
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button("📥 Download Trend Chart PNG", buf, file_name="traffic_trend_comparison.png", mime="image/png")

                st.sidebar.header("⚠️ Alerts")
                alert_drop_threshold = st.sidebar.number_input("Alert if drop % ≥", 0, 100, 30, step=1)
                alert_pages = drop_df[drop_df["Drop %"] >= alert_drop_threshold]

                if not alert_pages.empty:
                    st.sidebar.markdown(f"⚠️ Pages exceeding {alert_drop_threshold}% drop:")
                    for page in alert_pages[page_col].tolist():
                        st.sidebar.markdown(f"- {page}")
                else:
                    st.sidebar.markdown("No alerts currently.")

                missing_counts = df[date_cols].isna().sum().sum()
                if missing_counts > 0:
                    st.warning(f"⚠️ Your data contains {missing_counts} missing values, replaced with zero for calculations.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Please upload a CSV or Excel file with traffic data to start analysis.")
