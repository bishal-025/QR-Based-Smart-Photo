import streamlit as st
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="QR Smart Photo Organizer",
    page_icon="🎨",
    layout="wide"
)

# --- Custom Modern CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .category-header {
        font-size: 24px;
        font-weight: bold;
        color: #1E1E1E;
        padding: 10px 0px;
        border-bottom: 2px solid #4F46E5;
        margin-bottom: 15px;
    }
    .stExpander {
        border: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        background-color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- App Header ---
st.title("🖼️ QR Smart Photo Organizer")
st.markdown("Automated AI-driven organization for **Events, Education, and Hospitality**.")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/107/107175.png", width=80)
    st.header("Control Center")
    uploaded_files = st.file_uploader(
        "Upload Photos",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    st.divider()
    st.subheader("Display Settings")
    grid_size = st.select_slider("Gallery Density", options=[2, 3, 4, 5], value=3)

    st.button("✨ Auto-Organize", type="primary", use_container_width=True)

# --- Main Dashboard Logic ---
if uploaded_files:
    # 1. Top Performance Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📸 Total", len(uploaded_files))
    m2.metric("🎓 College", "Detected")
    m3.metric("🍴 Restaurant", "Detected")
    m4.metric("🏫 Classroom", "Detected")

    st.divider()

    # 2. Main Tabbed View
    tab1, tab2 = st.tabs(["📂 Smart Categories", "🔍 All Assets"])

    with tab1:
        # --- Section: College Event ---
        st.markdown('<div class="category-header">🎓 College Events</div>', unsafe_allow_html=True)
        with st.expander("View Graduation, Fest, or Seminar Photos", expanded=True):
            # Mock filtering first 2 photos for UI demo
            cols = st.columns(grid_size)
            for i, file in enumerate(uploaded_files[:2]):
                with cols[i % grid_size]:
                    st.image(file, use_container_width=True, caption=f"Event_Shot_{i+1}.jpg")

        # --- Section: Restaurant ---
        st.markdown('<div class="category-header">🍴 Restaurant & Dining</div>', unsafe_allow_html=True)
        with st.expander("View Menu Scans, Ambience, and Receipts", expanded=False):
            if len(uploaded_files) > 2:
                cols = st.columns(grid_size)
                for i, file in enumerate(uploaded_files[2:4]):
                    with cols[i % grid_size]:
                        st.image(file, use_container_width=True, caption=f"Table_Order_{i+1}.jpg")
            else:
                st.caption("No restaurant photos identified yet.")

        # --- Section: Classroom ---
        st.markdown('<div class="category-header">🏫 Classroom & Lectures</div>', unsafe_allow_html=True)
        with st.expander("View Whiteboard Notes and Assignments", expanded=False):
            if len(uploaded_files) > 4:
                cols = st.columns(grid_size)
                for i, file in enumerate(uploaded_files[4:]):
                    with cols[i % grid_size]:
                        st.image(file, use_container_width=True, caption=f"Lecture_Notes_{i+1}.jpg")
            else:
                st.caption("No classroom photos identified yet.")

    with tab2:
        st.subheader("Complete Master Gallery")
        cols = st.columns(grid_size)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % grid_size]:
                st.image(file, use_container_width=True)

else:
    # --- Landing Page / Empty State ---
    st.info("👈 Please upload your photo batch in the sidebar to generate the smart galleries.")

    # Placeholder layout to show how attractive it will look
    c1, c2, c3 = st.columns(3)
    c1.info("🎓 **College**\n\nSorts fests, seminars, and IDs.")
    c2.success("🍴 **Restaurant**\n\nSorts menus, bills, and orders.")
    c3.warning("🏫 **Classroom**\n\nSorts whiteboards and labs.")

    st.image("https://images.unsplash.com/photo-1540575467063-178a50c2df87?auto=format&fit=crop&q=80&w=1000",
             caption="Example: Organized Event Photography", use_container_width=True)

# --- Footer Footer ---
st.markdown("---")
st.caption("Smart Organizer • Powered by QR Detection")