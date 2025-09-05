import streamlit as st
from huggingface_hub import hf_hub_download

st.title("🚀 Test Deploy Streamlit")

REPO_ID = "zahratalitha/sentimen"

try:
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    st.success(f"Berhasil download config: {config_path}")
except Exception as e:
    st.error(f"Gagal download: {e}")
