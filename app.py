import os
import tempfile

import joblib
import pandas as pd
import streamlit as st

from pipeline import (
    extract_text_from_image,
    clean_ingredient_text,
    load_knowledge_base,
    check_warnings,
    compute_health_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="NutriScan", page_icon="🥗")
st.title("🥗 NutriScan — AI Food Label Decoder")
st.caption("Personalized ingredient, allergen, and health-risk analysis for Indian packaged food")


@st.cache_resource
def load_models():
    classifier = joblib.load(os.path.join(BASE_DIR, "models", "ingredient_classifier.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
    return classifier, vectorizer


@st.cache_resource
def load_kb():
    return load_knowledge_base()


classifier, vectorizer = load_models()
kb = load_kb()

st.subheader("1. Your Health Profile")
allergies = st.multiselect(
    "Allergies",
    ["dairy", "gluten", "soy", "peanut", "tree_nut", "egg", "shellfish", "fish", "sulphites", "phenylketonuria"],
)
conditions = st.multiselect(
    "Health Conditions",
    ["diabetes", "hypertension", "asthma"],
)

st.subheader("2. Provide the Ingredient List")
mode = st.radio("Input method", ["Upload a photo", "Paste ingredient text"], horizontal=True)

raw_text = ""
if mode == "Upload a photo":
    uploaded_file = st.file_uploader("Upload a clear photo of the ingredient list", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:   # unique temp file
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            with st.spinner("Reading label..."):
                raw_text = extract_text_from_image(tmp_path)
        finally:
            os.remove(tmp_path)
        with st.expander("Raw OCR text (for debugging)"):
            st.write(raw_text)
else:
    raw_text = st.text_area("Paste ingredient text", height=180,
                            placeholder="Ingredients: Wheat flour, sugar, palm oil, ... May contain traces of nuts.")

if raw_text.strip():
    ingredients = clean_ingredient_text(raw_text)

    if not ingredients:
        st.warning("Couldn't detect any ingredients. Try a clearer photo or paste the text.")
    else:
        user_profile = {"allergies": allergies, "conditions": conditions}
        results = check_warnings(ingredients, kb, classifier, vectorizer, user_profile)
        score = compute_health_score(results)

        st.subheader("3. Warnings")
        shown = False
        for r in results:                                   # red: confirmed allergens
            if r["allergen_warning"]:
                shown = True
                st.error(f"**{r['ingredient']}** — confirmed allergen: {', '.join(r['allergen_warning'])}")
        for r in results:                                   # orange: possible / trace
            soft = []
            if r["possible_allergen_warning"]:
                soft.append(f"possible allergen ({', '.join(r['possible_allergen_warning'])}) — "
                            "depends on source, label may not say")
            if r["trace_allergen_warning"]:
                soft.append(f"may contain traces ({', '.join(r['trace_allergen_warning'])})")
            if soft:
                shown = True
                st.markdown(
                    f"<div style='background:#fff1e0;border-left:6px solid #ff8c00;padding:0.6rem 0.9rem;"
                    f"border-radius:4px;margin-bottom:0.5rem;color:#5a3000'>"
                    f"<b>{r['ingredient']}</b> — {'; '.join(soft)}</div>",
                    unsafe_allow_html=True,
                )
        for r in results:                                   # yellow: conditions
            if r["condition_warning"]:
                shown = True
                st.warning(f"**{r['ingredient']}** — condition risk: {', '.join(r['condition_warning'])}")
        if not shown:
            st.success("No warnings for the allergies/conditions you selected.")

        st.subheader("4. Ingredient Details")
        table = pd.DataFrame([{
            "Ingredient": r["ingredient"] + (" (may contain)" if r["trace"] else ""),
            "Category": r["category"],
            "INS code": r["ins_code"] or "",
            "Risk": r["risk_level"],
            "Source": r["source"],
            "Notes": r["notes"],
        } for r in results])
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.subheader("5. Health Score")
        st.metric("Health Score", f"{score} / 100")
        st.caption("Starts at 100; minus risk points (medium 5, high 15, unknown 3), 20 per confirmed "
                   "allergen hit, 10 per possible/trace hit, 15 per condition hit — only for what you selected.")
