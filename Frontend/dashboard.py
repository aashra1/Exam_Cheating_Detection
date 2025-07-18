import sys
import streamlit as st
import pandas as pd
import os
from PIL import Image
import plotly.express as px
from datetime import datetime
import io
from base64 import b64encode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection import face_detection, object_detection, pose_detection
from utils import cheating_logic, tracker
from utils.detection_helpers import compute_iou, merge_pose_to_tracked
from Backend import db


def get_logs_from_db():
    try:
        logs_cursor = db.logs_collection.find().sort("timestamp", -1)
        logs_list = list(logs_cursor)
        if not logs_list:
            return pd.DataFrame(columns=["timestamp", "class_id", "face_id", "activity", "severity", "image_path", "video_url"])
        records = []
        for log in logs_list:
            records.append({
                "timestamp": log.get("timestamp"),
                "class_id": log.get("class_id"),
                "face_id": log.get("face_id"),
                "activity": log.get("activity"),
                "severity": log.get("severity"),
                "image_path": log.get("image_url"),
                "video_url": log.get("video_url")
            })
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        st.error(f"Error fetching logs from MongoDB: {e}")
        return pd.DataFrame(columns=["timestamp", "class_id", "face_id", "activity", "severity", "image_path", "video_url"])


def format_severity(sev):
    if sev == "warning":
        return "🟡 Warning"
    elif sev == "critical":
        return "🔴 Critical"
    else:
        return sev.title()


def dashboard():
    st.set_page_config(page_title="Cheating Detection", layout="wide")

    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            [data-testid="stSidebar"] {
                background: #1e293b !important;
                border-right: 1px solid #334155;
            }
            div[role="radiogroup"] > label {
                padding: 0.75rem 1.5rem !important;
                margin: 0.25rem 0 !important;
                color: #cbd5e1 !important;
                font-weight: 500 !important;
                border-radius: 0 !important;
                display: flex !important;
                align-items: center !important;
            }
            div[role="radiogroup"] > label:hover {
                background: #334155 !important;
                color: white !important;
            }
            div[role="radiogroup"] > label[data-baseweb="radio"]:has(> div[aria-checked="true"]) {
                background: #3b82f6 !important;
                color: white !important;
                font-weight: 600 !important;
            }
            div[role="radiogroup"] > label > div:first-child {
                display: none !important;
            }
            [data-testid="stSidebar"] * {
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

    df = get_logs_from_db()

    with st.sidebar:
        # Combined Logo + Title
        logo_path = "Frontend/assets/logo.png"
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_data = f.read()
            logo_base64 = b64encode(logo_data).decode()
            st.markdown(f'''
                       <div style="text-align: center; margin-top: -2rem;">
            <img src="data:image/png;base64,{logo_base64}" width="400" style="margin: auto; display: block;" />
        </div>
        <hr style="border: none; height: 2px; background-color: white; margin: 0 0 1rem 0; width: 100%;">


            ''', unsafe_allow_html=True)
        else:
            st.warning("Logo image not found at: Frontend/assets/logo.png")

        # Navigation
        page = st.radio(
            "Navigation",
            ["Activity Logs", "Flagged Snapshots", "Video Clips", "Download Logs", "Summary"],
            label_visibility="collapsed",
        )

        # Status box
        st.markdown("""
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div style="color: white; font-size: 0.9rem; margin-bottom: 0.5rem;">System Status</div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; background: #10b981; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="color: white; font-size: 0.9rem;">Active</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ---------------- Main content pages ----------------

    if page == "Activity Logs":
        st.header("Activity Logs")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            severity_filter = st.selectbox("Severity", ["All", "warning", "critical"])
        with col2:
            class_ids = ["All"] + sorted(df["class_id"].dropna().astype(str).unique().tolist())
            class_filter = st.selectbox("Class ID", class_ids)
        with col3:
            date_range = st.date_input("Date range", [df["timestamp"].min().date(), df["timestamp"].max().date()])
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = df["timestamp"].min().date()

        filtered_df = df.copy()
        if severity_filter != "All":
            filtered_df = filtered_df[filtered_df["severity"] == severity_filter]
        if class_filter != "All":
            filtered_df = filtered_df[filtered_df["class_id"].astype(str) == class_filter]
        filtered_df = filtered_df[
            (filtered_df["timestamp"].dt.date >= start_date) &
            (filtered_df["timestamp"].dt.date <= end_date)
        ]

        display_df = filtered_df.copy()
        display_df["severity"] = display_df["severity"].apply(format_severity)
        st.dataframe(
            display_df[["timestamp", "class_id", "face_id", "activity", "severity"]],
            use_container_width=True,
            hide_index=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Incidents", len(df))
        col2.metric("Critical", len(df[df["severity"] == "critical"]))
        col3.metric("Warnings", len(df[df["severity"] == "warning"]))

    elif page == "Flagged Snapshots":
        st.header("Flagged Snapshots")

        if df.empty:
            st.info("No snapshot data.")
        else:
            min_date = df["timestamp"].min().date()
            max_date = df["timestamp"].max().date()
            date_range = st.date_input("Filter snapshots by Date Range", [min_date, max_date])
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = min_date

            filtered_snapshots = df[
                (df["image_path"].notna()) &
                (df["image_path"].str.startswith("http")) &
                (df["timestamp"].dt.date >= start_date) &
                (df["timestamp"].dt.date <= end_date)
            ]

            for i, row in filtered_snapshots.iterrows():
                with st.expander(f"📸 {row['timestamp']} | Face {row['face_id']}"):
                    st.image(row["image_path"], width=200)
                    st.write(f"*Activity*: {row['activity']}")
                    st.write(f"*Severity*: {format_severity(row['severity'])}")

    elif page == "Video Clips":
        st.header("Flagged Video Clips")

        if df.empty:
            st.info("No video clips.")
        else:
            min_date = df["timestamp"].min().date()
            max_date = df["timestamp"].max().date()
            date_range = st.date_input("Filter videos by Date Range", [min_date, max_date])
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = min_date

            filtered_videos = df[
                (df["video_url"].notna()) &
                (df["video_url"] != "") &
                (df["timestamp"].dt.date >= start_date) &
                (df["timestamp"].dt.date <= end_date)
            ]

            if filtered_videos.empty:
                st.info("No video clips in this date range.")
            else:
                for _, row in filtered_videos.iterrows():
                    with st.expander(f"🎥 {row['timestamp']} | Face {row['face_id']}"):
                        st.video(row["video_url"])
                        st.write(f"*Activity*: {row['activity']}")

    elif page == "Download Logs":
        st.header("Download Logs")

        export_scope = st.radio("Export", ["All Logs", "Filtered Logs"])
        export_format = st.radio("Format", ["CSV", "Excel"], horizontal=True)

        data_to_export = df if export_scope == "All Logs" else filtered_df

        if not data_to_export.empty:
            if export_format == "CSV":
                data = data_to_export.to_csv(index=False).encode("utf-8")
                mime = "text/csv"
                filename = "cheating_logs.csv"
            else:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    data_to_export.to_excel(writer, index=False)
                data = buffer.getvalue()
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                filename = "cheating_logs.xlsx"

            st.download_button("📥 Download Logs", data=data, file_name=filename, mime=mime)
        else:
            st.info("No data available for export.")

    elif page == "Summary":
        st.header("Detection Summary")

        if not df.empty:
            activity_counts = df["activity"].value_counts().reset_index()
            activity_counts.columns = ["activity", "count"]
            fig1 = px.bar(
                activity_counts,
                x="activity",
                y="count",
                color="activity",
                labels={"activity": "Activity", "count": "Count"},
                title="Activity Frequency",
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("⚠ Severity Breakdown")
            fig2 = px.pie(
                df,
                names="severity",
                title="Severity Split",
                color_discrete_map={"warning": "#facc15", "critical": "#ef4444"},
                hole=0.3,
            )
            st.plotly_chart(fig2)
        else:
            st.info("No data to summarize.")


if __name__ == "__main__":
    dashboard()
