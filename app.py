import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Analisis Sentimen", page_icon="🧠")
st.title("🧠 Analisis Sentimen Komentar Kasus Tom Lembong")

# ID repo model di Hugging Face
REPO_ID = "zahratalitha/sentimen"

# Mapping label ke nama custom
id2label = {
    "LABEL_0": "SADNESS",
    "LABEL_1": "ANGER",
    "LABEL_2": "SUPPORT",
    "LABEL_3": "HOPE",
    "LABEL_4": "DISAPPOINTMENT",
}

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model=REPO_ID,
        tokenizer=REPO_ID,
        return_all_scores=False
    )

nlp = load_model()

# Input user
user_text = st.text_area("Masukkan teks komentar:", height=150)

examples = [
    "Tom Lembong dituding melakukan pelanggaran, publik merasa kecewa dengan sikapnya.",
    "Saya mendukung Tom Lembong karena beliau jujur.",
    "Publik marah besar atas tindakan yang dilakukan.",
    "Masih ada harapan agar kasus ini diselesaikan dengan baik.",
]

st.write("Contoh cepat:")
for ex in examples:
    if st.button(ex):
        user_text = ex

if st.button("Prediksi"):
    if user_text.strip():
        result = nlp(user_text)[0]   # ambil hasil top-1
        label = id2label.get(result["label"], result["label"])
        st.success(f"Label: **{label}** ({result['score']:.2%})")
        st.caption(f"Teks: `{user_text}`")
    else:
        st.warning("Tolong masukkan teks terlebih dahulu.")
