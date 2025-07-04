import sys
import streamlit as st
import pandas as pd
import os
from PIL import Image
import cv2
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection import face_detection, object_detection, pose_detection
from utils import cheating_logic, tracker
from utils.detection_helpers import compute_iou, merge_pose_to_tracked
from Backend import db  # PostgreSQL connector

# ------------------- Helper: Fetch Logs --------------------
def get_logs_from_db():
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, class_id, face_id, activity, severity, image_url, video_url 
            FROM logs ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        df = pd.DataFrame(rows, columns=["timestamp", "class_id", "face_id", "activity", "severity", "image_path", "video_url"])
        return df
    except Exception as e:
        st.error(f"Error fetching logs from database: {e}")
        return pd.DataFrame(columns=["timestamp", "class_id", "face_id", "activity", "severity", "image_path", "video_url"])

# ------------------- Main Dashboard --------------------
def dashboard():
    # ========== CSS Styling ==========
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background: #1e293b !important;
                border-right: 1px solid #334155;
            }
            .sidebar-title {
                color: white !important;
                font-size: 1.5rem !important;
                font-weight: 700;
                padding: 1rem;
                border-bottom: 1px solid #334155;
                margin-bottom: 1rem;
            }
            div[role="radiogroup"] > label {
                padding: 0.75rem 1.5rem !important;
                margin: 0.25rem 0 !important;
                color: #cbd5e1 !important;
                font-weight: 500 !important;
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
            .status-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 12px;
                font-size: 0.85rem;
                font-weight: 500;
            }
            .warning {
                background: #fef3c7;
                color: #92400e;
            }
            .critical {
                background: #fee2e2;
                color: #991b1b;
            }
            .white-card {
                background-color: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                margin-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # ========== Sidebar ==========
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Cheating Detection</div>', unsafe_allow_html=True)

        page = st.radio("Navigation", ["Activity Logs", "Flagged Snapshots", "Video Clips", "Download Logs", "Summary"], label_visibility="collapsed")

        st.markdown("""
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">System Status</div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; background: #10b981; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="color: white; font-size: 0.9rem;">Active</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    df = get_logs_from_db()

    # ========== Activity Logs ==========
    if page == "Activity Logs":
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.header("Activity Logs")

        col1, col2, col3 = st.columns(3)
        with col1:
            severity_filter = st.selectbox("Filter by Severity", ["All", "warning", "critical"])
        with col2:
            class_filter = st.selectbox("Filter by Class ID", ["All"] + sorted(df["class_id"].dropna().unique()))
        with col3:
            date_filter = st.date_input("Filter by Date")

        filtered_df = df.copy()

        if severity_filter != "All":
            filtered_df = filtered_df[filtered_df["severity"] == severity_filter]
        if class_filter != "All":
            filtered_df = filtered_df[filtered_df["class_id"] == class_filter]
        if isinstance(date_filter, pd._libs.tslibs.timestamps.Timestamp):
            filtered_df = filtered_df[pd.to_datetime(filtered_df["timestamp"]).dt.date == date_filter]

        def format_severity(sev):
            cls = "warning" if sev == "warning" else "critical"
            return f'<span class="status-badge {cls}">{sev.title()}</span>'

        display_df = filtered_df.copy()
        display_df["severity"] = display_df["severity"].apply(format_severity)

        st.dataframe(
            display_df[["timestamp", "class_id", "face_id", "activity", "severity"]],
            use_container_width=True,
            hide_index=True,
            column_config={"severity": st.column_config.Column("Severity")}
        )

        st.subheader("Detection Statistics")
        cols = st.columns(3)
        cols[0].metric("Total Incidents", len(filtered_df))
        cols[1].metric("Critical", len(filtered_df[filtered_df["severity"] == "critical"]))
        cols[2].metric("Warnings", len(filtered_df[filtered_df["severity"] == "warning"]))
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Snapshots ==========
    elif page == "Flagged Snapshots":
        st.header("Flagged Snapshots")
        if df.empty:
            st.info("No logs found to display snapshots.")
        else:
            cols = st.columns(2)
            for i, row in df.iterrows():
                img_url = row["image_path"]
                if img_url and img_url.startswith("http"):
                    with cols[i % 2]:
                        st.image(img_url, use_column_width=True)
                        st.caption(f"{row['timestamp']} - {row['activity']}")

    # ========== Videos ==========
    elif page == "Video Clips":
        st.header("Flagged Video Clips")
        if df.empty:
            st.info("No logs found to display videos.")
        else:
            video_df = df[df["video_url"].notna() & (df["video_url"] != "")]
            for i, row in video_df.iterrows():
                st.video(row["video_url"])
                st.caption(f"{row['timestamp']} | {row['activity']}")

    # ========== Download ==========
    elif page == "Download Logs":
        st.header("Download Logs")
        st.radio("Export Format", ["CSV", "Excel"], horizontal=True, key="export_format")
        if not df.empty:
            csv = df.to_csv(index=False).encode('utf-8')
            import io
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            excel_data = excel_buffer.getvalue()

            st.download_button(
                "Download Data",
                csv if st.session_state.export_format == "CSV" else excel_data,
                "cheating_logs.csv" if st.session_state.export_format == "CSV" else "cheating_logs.xlsx",
                "text/csv" if st.session_state.export_format == "CSV" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("No logs to export.")

    # ========== Summary ==========
    elif page == "Summary":
        st.header("Detection Summary")
        if not df.empty:
            st.subheader("Activity Distribution")
            st.bar_chart(df["activity"].value_counts())

            st.subheader("Severity Breakdown")
            st.pyplot(
                df["severity"].value_counts().plot.pie(
                    autopct='%1.1f%%',
                    colors=["#fef3c7", "#fee2e2"],
                    labels=["warning", "critical"]
                ).figure
            )
        else:
            st.info("No summary data available.")

if __name__ == "__main__":
    dashboard()
