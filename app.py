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
        st.warning(
            "Couldn't reliably identify the ingredient section. "
            "Please upload a clearer photo showing the word "
            "'Ingredients' and the complete ingredient list."
        )

    else:
        st.subheader("Extracted ingredient candidates")
        st.write(ingredients)

        user_profile = {
            "allergies": allergies,
            "conditions": conditions,
        }

        results = check_warnings(
            ingredients,
            kb,
            classifier,
            vectorizer,
            user_profile,
        )

        st.subheader("3. Results")

        # Unknown ingredients and missing allergen information
        # must not be treated as proof of safety.
        review_needed = any(
            r.get("review_required", True)
            for r in results
        )

        if review_needed:
            st.warning(
                "Some ingredients are unmatched or lack recorded "
                "allergen information. Results may be incomplete; "
                "please review the label manually."
            )

        # Show the score only when every ingredient was matched
        # and the current warning results do not require review.
        if not review_needed:
            score = compute_health_score(results)
            st.metric("Health Score", f"{score} / 100")
        else:
            st.metric("Health Score", "Unavailable")
            st.caption(
                "A reliable score cannot be displayed until "
                "the ingredient information has been reviewed."
            )

        for r in results:
            line = (
                f"**{r['ingredient']}** — "
                f"_{r['category']}_ "
                f"(risk: {r['risk_level']})"
            )

            if r["allergen_warning"]:
                st.error(
                    f"{line}\n\n"
                    f"⚠️ Allergen match: "
                    f"{', '.join(r['allergen_warning'])}"
                )

            elif r["condition_warning"]:
                st.warning(
                    f"{line}\n\n"
                    f"⚠️ Condition warning: "
                    f"{', '.join(r['condition_warning'])}"
                )

            else:
                st.markdown(line)

            if r.get("review_required", True):
                st.caption(
                    "⚠️ Manual review required: ingredient matching "
                    "or allergen information is incomplete."
                )