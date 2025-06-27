import streamlit as st

# ------------------ APP CONFIG ------------------

st.set_page_config(page_title="Cheating Detection", layout="centered")

# ------------------ SESSION STATE ------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user_data" not in st.session_state:
    st.session_state.user_data = {}  # Store fake user credentials temporarily

# ------------------ LOGIN PAGE ------------------

def login():
    st.markdown("<h2 style='text-align: center;'>🔐 Secure Login</h2>", unsafe_allow_html=True)
    st.markdown("#### Welcome back! Please enter your credentials to continue.")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if username and password:
                stored_password = st.session_state.user_data.get(username)
                if stored_password and stored_password == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"✅ Welcome, {username}!")
                    # No dashboard to reroute to
                else:
                    st.error("❌ Invalid username or password.")
            else:
                st.warning("Please fill in both fields.")

# ------------------ SIGN UP PAGE ------------------

def signup():
    st.markdown("<h2 style='text-align: center;'>📝 Create a New Account</h2>", unsafe_allow_html=True)
    st.markdown("#### Just a few quick steps to get started.")

    with st.form("signup_form"):
        username = st.text_input("Choose a Username", placeholder="Create a unique username")
        password = st.text_input("Choose a Password", type="password", placeholder="Create a secure password")
        submitted = st.form_submit_button("Sign Up", use_container_width=True)

        if submitted:
            if username and password:
                if username in st.session_state.user_data:
                    st.error("❌ Username already exists.")
                else:
                    st.session_state.user_data[username] = password
                    st.success("✅ Account created! You can now log in.")
                    st.session_state.page = "login"
                    st.rerun()
            else:
                st.warning("Both fields are required.")

# ------------------ ROUTING ------------------

if st.session_state.page == "login":
    login()
    st.markdown("---")
    if st.button("👉 Create New Account", use_container_width=True):
        st.session_state.page = "signup"
        st.rerun()
elif st.session_state.page == "signup":
    signup()
    st.markdown("---")
    if st.button("🔙 Back to Login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()
