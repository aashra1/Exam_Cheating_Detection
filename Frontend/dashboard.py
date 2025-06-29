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

# Import your db helper to fetch logs
from utils import db  # assuming your db.py has a function to fetch logs from DB

def get_logs_from_db():
    # Fetch all logs from your DB table `logs`
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, class_id, face_id, activity, severity, image_url FROM logs ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        # Convert to pandas DataFrame with matching columns
        df = pd.DataFrame(rows, columns=["timestamp", "class_id", "face_id", "activity", "severity", "image_path"])
        return df
    except Exception as e:
        st.error(f"Error fetching logs from database: {e}")
        return pd.DataFrame(columns=["timestamp", "class_id", "face_id", "activity", "severity", "image_path"])

def dashboard():
    # ===== STYLING =====
    st.markdown("""
        <style>
            /* Main layout */
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            
            /* Sidebar styling */
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
            
            /* Navigation items */
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
            
            /* Hide radio buttons */
            div[role="radiogroup"] > label > div:first-child {
                display: none !important;
            }
            
            /* Content cards */
            .card {
                background: transparent !important;
                box-shadow: none !important;
                border: none !important;
            }
            
            /* Headers */
            h1 {
                font-size: 1.5rem;
                color: #1e293b;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #e2e8f0;
            }
            
            /* Tables */
            div[data-testid="stDataFrameContainer"] {
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
            
            /* Buttons */
            button[kind="primary"] {
                background: #3b82f6 !important;
                border: none !important;
            }
            
            /* Status badges */
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
        </style>
    """, unsafe_allow_html=True)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Cheating Detection</div>', unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Activity Logs", "Flagged Snapshots", "Download Logs", "Summary"],
            label_visibility="collapsed",
        )

        st.markdown("""
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">System Status</div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; background: #10b981; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="color: white; font-size: 0.9rem;">Active</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Fetch logs from DB here instead of CSV
    df = get_logs_from_db()

    # ===== MAIN CONTENT =====
    with st.container():
        if page == "Activity Logs":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("Activity Logs")

            col1, _ = st.columns(2)
            with col1:
                severity_filter = st.selectbox("Filter by severity", ["All", "warning", "critical"])

            if severity_filter == "warning":
                filtered_df = df[df["severity"] == "warning"]
            elif severity_filter == "critical":
                filtered_df = df[df["severity"] == "critical"]
            else:
                filtered_df = df

            def format_severity(sev):
                cls = "warning" if sev == "warning" else "critical"
                return f'<span class="status-badge {cls}">{sev.title()}</span>'

            display_df = filtered_df.copy()
            display_df["severity"] = display_df["severity"].apply(format_severity)

            st.dataframe(
                display_df[["timestamp", "class_id", "face_id", "activity", "severity"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "severity": st.column_config.Column("Severity")
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detection Statistics")
            cols = st.columns(3)
            cols[0].metric("Total Incidents", len(df))
            cols[1].metric("Critical", len(df[df["severity"] == "critical"]))
            cols[2].metric("Warnings", len(df[df["severity"] == "warning"]))
            st.markdown('</div>', unsafe_allow_html=True)

        elif page == "Flagged Snapshots":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("Flagged Snapshots")

            if df.empty:
                st.info("No logs found to display snapshots.")
            else:
                cols = st.columns(2)
                for i, row in df.iterrows():
                    img_path = row["image_path"]
                    if img_path and os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            with cols[i % 2]:
                                st.image(img, use_column_width=True)
                                st.caption(f"{row['timestamp']} - {row['activity']}")
                        except Exception as e:
                            st.error(f"Error loading image {img_path}: {e}")
                    else:
                        continue

            st.markdown('</div>', unsafe_allow_html=True)

        elif page == "Download Logs":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("Download Logs")

            st.radio("Export format", ["CSV", "Excel"], horizontal=True, key="export_format")

            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                try:
                    import io
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    excel_data = excel_buffer.getvalue()
                except Exception:
                    excel_data = None

                st.download_button(
                    "Download Data",
                    csv if st.session_state.export_format == "CSV" else excel_data,
                    "cheating_logs.csv" if st.session_state.export_format == "CSV" else "cheating_logs.xlsx",
                    "text/csv" if st.session_state.export_format == "CSV" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("No log data available to download.")

            st.markdown('</div>', unsafe_allow_html=True)

        elif page == "Summary":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("Detection Summary")

            if df.empty or df["activity"].empty:
                st.info("No activity data available for chart.")
            else:
                st.subheader("Activity Distribution")
                st.bar_chart(df["activity"].value_counts())

            if df.empty or df["severity"].empty:
                st.info("No severity data available for chart.")
            else:
                st.subheader("Severity Breakdown")
                st.pyplot(
                    df["severity"].value_counts().plot.pie(
                        autopct='%1.1f%%',
                        colors=["#fef3c7", "#fee2e2"],
                        labels=["warning", "critical"]
                    ).figure
                )
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    dashboard()
