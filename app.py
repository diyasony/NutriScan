import streamlit as st
import joblib
from pipeline import (
    extract_text_from_image,
    clean_ingredient_text,
    load_knowledge_base,
    check_warnings,
    compute_health_score,
)

st.set_page_config(page_title="NutriScan", page_icon="🥗")
st.title("🥗 NutriScan — AI Food Label Decoder")
st.caption("Personalized ingredient, allergen, and health-risk analysis for Indian packaged food")

classifier = joblib.load("models/ingredient_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
kb = load_knowledge_base()

st.subheader("1. Your Health Profile")
allergies = st.multiselect(
    "Allergies",
    ["dairy", "gluten", "soy", "peanut", "tree_nut", "egg", "shellfish", "fish", "sulphites", "phenylketonuria"],
)
conditions = st.multiselect(
    "Health Conditions",
    ["diabetes", "hypertension", "asthma"],
)

st.subheader("2. Upload a Food Label Photo")
uploaded_file = st.file_uploader("Upload a clear photo of the ingredient list", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("temp_label.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading label..."):
        raw_text = extract_text_from_image("temp_label.jpg")

    with st.expander("Raw OCR text (for debugging)"):
        st.write(raw_text)

    ingredients = clean_ingredient_text(raw_text)

    if not ingredients:
        st.warning("Couldn't detect any ingredients. Try a clearer photo.")
    else:
        user_profile = {"allergies": allergies, "conditions": conditions}
        results = check_warnings(ingredients, kb, classifier, vectorizer, user_profile)
        score = compute_health_score(results)

        st.subheader("3. Results")
        st.metric("Health Score", f"{score} / 100")

        for r in results:
            line = f"**{r['ingredient']}** — _{r['category']}_ (risk: {r['risk_level']})"
            if r["allergen_warning"]:
                st.error(f"{line}  \n⚠️ Allergen match: {', '.join(r['allergen_warning'])}")
            elif r["condition_warning"]:
                st.warning(f"{line}  \n⚠️ Condition risk: {', '.join(r['condition_warning'])}")
            else:
                st.markdown(line)
