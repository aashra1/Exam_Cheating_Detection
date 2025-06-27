import streamlit as st
import pandas as pd
import os
from PIL import Image

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
                
            background: transparent !important;  /* Remove white background */
            box-shadow: none !important;       /* Remove shadow */
            border: none !important;           /* Remove border */  
                
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
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["Activity Logs", "Flagged Snapshots", "Download Logs", "Summary"],
            label_visibility="collapsed",
            
        )
        
        # Status indicator
        st.markdown("""
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">System Status</div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; background: #10b981; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="color: white; font-size: 0.9rem;">Active</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ===== MAIN CONTENT =====
    # Sample data
    fake_data = {
        "timestamp": ["2025-06-25 14:01:23", "2025-06-25 14:15:47", "2025-06-25 14:32:11"],
        "student_id": ["S101", "S103", "S105"],
        "activity": [
            "Looking at another student's paper", 
            "Using phone during exam",
            "Whispering to neighbor"
        ],
        "severity": ["critical", "critical", "warning"],
        "image_path": ["snapshots/incident1.jpg", "snapshots/incident2.jpg", "snapshots/incident3.jpg"]
    }
    df = pd.DataFrame(fake_data)

    with st.container():
        if page == "Activity Logs":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("Activity Logs")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                severity_filter = st.selectbox("Filter by severity", ["All", "Warning", "Critical"])
            
            # Apply filters
            if severity_filter == "Warning":
                filtered_df = df[df["severity"] == "warning"]
            elif severity_filter == "Critical":
                filtered_df = df[df["severity"] == "critical"]
            else:
                filtered_df = df
            
            # Format severity as badges
            def format_severity(severity):
                cls = "warning" if severity == "warning" else "critical"
                return f'<span class="status-badge {cls}">{severity.title()}</span>'
            
            display_df = filtered_df.copy()
            display_df["severity"] = display_df["severity"].apply(format_severity)
            
            st.dataframe(
                display_df[["timestamp", "student_id", "activity", "severity"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "severity": st.column_config.Column("Severity")
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Stats cards
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
            
            img_dir = "snapshots"
            if os.path.exists(img_dir):
                images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
                if images:
                    cols = st.columns(2)
                    for i, img_name in enumerate(images):
                        img_path = os.path.join(img_dir, img_name)
                        try:
                            img = Image.open(img_path)
                            with cols[i % 2]:
                                st.image(img, use_column_width=True)
                                st.caption(img_name)
                        except Exception as e:
                            st.error(f"Error loading {img_name}: {str(e)}")
                else:
                    st.info("No images found in snapshots directory")
            else:
                st.warning("Snapshots directory not found")
            st.markdown('</div>', unsafe_allow_html=True)

        elif page == "Download Logs":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("Download Logs")
            
            # Export options
            st.radio("Export format", ["CSV", "Excel"], horizontal=True, key="export_format")
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Data",
                csv if st.session_state.export_format == "CSV" else df.to_excel(index=False),
                "cheating_logs.csv" if st.session_state.export_format == "CSV" else "cheating_logs.xlsx",
                "text/csv" if st.session_state.export_format == "CSV" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        elif page == "Summary":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("Detection Summary")
            
            # Activity distribution
            st.subheader("Activity Distribution")
            st.bar_chart(df["activity"].value_counts())
            
            # Severity breakdown
            st.subheader("Severity Breakdown")
            st.pyplot(df["severity"].value_counts().plot.pie(
                autopct='%1.1f%%',
                colors=["#fef3c7", "#fee2e2"],
                labels=["Warning", "Critical"]
            ).figure)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    dashboard()