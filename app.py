"""
NutriScan - Streamlit front-end.

Contract with pipeline.py (unchanged): extract_text_from_image, clean_ingredient_text,
load_knowledge_base, check_warnings, compute_health_score.
Run:  streamlit run app.py
Everything works offline: system fonts, inline CSS, no extra dependencies.
"""
import html
import os
import tempfile

import joblib
import pandas as pd
import streamlit as st

import pipeline
from pipeline import (check_warnings, clean_ingredient_text, compute_health_score,
                      extract_text_from_image, load_knowledge_base)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALLERGENS = [("dairy", "Dairy"), ("gluten", "Gluten"), ("soy", "Soy"), ("peanut", "Peanut"),
             ("tree_nut", "Tree nut"), ("egg", "Egg"), ("shellfish", "Shellfish"), ("fish", "Fish"),
             ("sulphites", "Sulphites"), ("phenylketonuria", "Phenylketonuria (PKU)")]
CONDITIONS = [("diabetes", "Diabetes"), ("hypertension", "Hypertension"), ("asthma", "Asthma")]
ALLERGEN_LABEL = dict(ALLERGENS)
CONDITION_LABEL = dict(CONDITIONS)

# Verdict thresholds are a UI design choice (ASSUMED, not from the proposal).
AVOID_BELOW = 40
CAUTION_BELOW = 70

DISCLAIMER = ("Allergen and risk tags are team-assigned rules, not medical advice; "
              "additive list derived from FSSAI documents.")

# ----------------------------------------------------------------------------- sample labels
# Typed by the team for demo purposes. They are NOT copied from real products.
SAMPLES = {
    "snickers": {
        "button": "🍫 Peanut chocolate bar",
        "title": "Peanut chocolate bar (typed sample)",
        "text": ("Ingredients: Milk chocolate (sugar, cocoa butter, skimmed milk powder, cocoa mass, "
                 "lactose, emulsifier (soy lecithin)), peanuts (21%), sugar, glucose syrup, palm oil, "
                 "iodised salt, egg white, flavouring substances (vanillin).\n"
                 "Contains Milk, Peanut and Soy. May contain traces of tree nuts.\n"
                 "Nutritional Information per 100 g\nEnergy 488 kcal\nProtein 9 g\nTotal Fat 24 g"),
    },
    "noodles": {
        "button": "🍜 Instant noodles",
        "title": "Instant noodles (typed sample)",
        "text": ("INGREDIENTS\nNoodles: Wheat flour, palm oil, salt, wheat gluten, thickener (INS 412), "
                 "acidity regulators (INS 501(i)), mineral (calcium carbonate), antioxidant (INS 319).\n"
                 "Tastemaker: Spices (onion powder, garlic powder, chilli powder, black pepper), sugar, "
                 "salt, hydrolysed soy protein, flavour enhancers (INS 627, INS 631), maltodextrin, "
                 "colour (INS 150d), anticaking agent (INS 551).\n"
                 "Contains: Wheat, Soy. Produced in a facility that also handles peanuts, milk and eggs.\n"
                 "Nutrition information per serve\nProtein 8 g, Sodium 1100 mg\nNet weight 70 g  MRP Rs 15"),
    },
    "biscuit": {
        "button": "🍪 Biscuit",
        "title": "Biscuit (typed sample)",
        "text": ("INGREDIENTS: Refined wheat flour (maida) (58%), sugar, edible vegetable oil (palm),\n"
                 "invert syrup, milk solids, leavening agents (INS 500(ii), INS 503(ii)), iodised salt,\n"
                 "emulsifiers (soy lecithin, INS 322(i)), added flavours (artificial flavouring "
                 "substances - vanilla).\nContains wheat, milk and soy. May contain traces of nuts.\n"
                 "NUTRITIONAL INFORMATION (approx. values per 100 g)\nEnergy 480 kcal\nProtein 6.5 g\n"
                 "Total Fat 18 g\nSodium 320 mg"),
    },
    "mayo": {
        "button": "🥚 Veg mayonnaise",
        "title": "Veg mayonnaise (typed sample, Veeba-style)",
        "text": ("Ingredients: Refined soyabean oil, water, sugar, milk solids, iodised salt, "
                 "synthetic vinegar, mustard, stabilizers (INS 412, INS 415), "
                 "preservative (INS 211), acidity regulator (INS 330), spices and condiments.\n"
                 "Contains Milk and Soy.\nStorage: Refrigerate after opening."),
    },
}

# ----------------------------------------------------------------------------- CSS
CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 820px;}
.ns-hero {background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 55%, #43a047 100%); color: #fff;
  border-radius: 18px; padding: 22px 22px 18px 22px; margin-bottom: 14px;
  box-shadow: 0 4px 14px rgba(27,94,32,.25);}
.ns-hero h1 {color: #fff; margin: 0; font-size: 2rem; line-height: 1.15;}
.ns-hero p {color: #e8f5e9; margin: 6px 0 0 0; font-size: 1rem;}
.ns-chips span {display: inline-block; background: rgba(255,255,255,.18); color: #fff;
  border-radius: 999px; padding: 2px 10px; margin: 8px 6px 0 0; font-size: .78rem;}
.ns-verdict {border-radius: 18px; padding: 18px 20px; margin: 10px 0 14px 0; border: 2px solid;}
.ns-verdict .big {font-size: 1.7rem; font-weight: 800; margin: 0;}
.ns-verdict .sub {margin: 4px 0 0 0; font-size: .95rem; color: #374151;}
.ns-ok {background: #e8f6ec; border-color: #2e9d4f; color: #14532d;}
.ns-caution {background: #fff4e0; border-color: #f59e0b; color: #7c4a03;}
.ns-avoid {background: #fdecea; border-color: #d32f2f; color: #7f1d1d;}
.ns-neutral {background: #eef2f7; border-color: #94a3b8; color: #334155;}
.ns-scorewrap {display: flex; align-items: center; gap: 14px; margin-top: 12px;}
.ns-score {font-size: 2.2rem; font-weight: 800; min-width: 92px; line-height: 1;}
.ns-score small {font-size: .9rem; font-weight: 600; color: #4b5563;}
.ns-bar {flex: 1; height: 16px; background: rgba(0,0,0,.10); border-radius: 999px; overflow: hidden;}
.ns-fill {height: 100%; border-radius: 999px;}
.ns-card {border-radius: 12px; padding: 10px 14px; margin: 8px 0; border-left: 8px solid; color: #1f2937;}
.ns-card b {color: inherit;}
.ns-card .why {display: block; font-size: .82rem; color: #4b5563; margin-top: 2px;}
.ns-red {background: #fdecea; border-color: #d32f2f;}
.ns-orange {background: #fff1e0; border-color: #f57c00;}
.ns-yellow {background: #fffbe0; border-color: #e6b800;}
.ns-green {background: #e8f6ec; border-color: #2e9d4f;}
.ns-scroll {overflow-x: auto;}
table.ns-table {border-collapse: collapse; width: 100%; font-size: .88rem; color: #1f2937;}
table.ns-table th {text-align: left; background: #e8f0ea; padding: 7px 9px; white-space: nowrap;}
table.ns-table td {padding: 7px 9px; border-bottom: 1px solid #e5e7eb; vertical-align: top;
  background: #fff;}
.ns-badge {display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: .75rem;
  font-weight: 700; white-space: nowrap;}
.b-low {background: #d9f2e0; color: #14532d;} .b-medium {background: #ffe3b3; color: #7c4a03;}
.b-high {background: #ffc9c4; color: #7f1d1d;} .b-unknown {background: #e5e7eb; color: #374151;}
.b-trace {background: #ffe0b2; color: #7c4a03;} .b-junk {background: #f3f4f6; color: #6b7280;}
.ns-foot {color: #6b7280; font-size: .8rem; text-align: center; margin-top: 26px;
  border-top: 1px solid #e5e7eb; padding-top: 12px;}
@media (max-width: 640px) {.ns-hero h1 {font-size: 1.55rem;} .ns-verdict .big {font-size: 1.4rem;}
  .ns-score {font-size: 1.8rem; min-width: 76px;}}
</style>
"""

# ----------------------------------------------------------------------------- pure helpers
def esc(x):
    return html.escape("" if x is None else str(x))


def nice_list(keys, labels):
    return ", ".join(labels.get(k, k) for k in keys)


def score_colour(score):
    if score >= CAUTION_BELOW:
        return "#2e9d4f"
    if score >= AVOID_BELOW:
        return "#f59e0b"
    return "#d32f2f"


def collect_warnings(results):
    """Warning cards as (level, title, detail), level in red / orange / yellow.
    Built only from what check_warnings reported."""
    cards = []
    declares = list(getattr(results, "label_declares", ()) or ())
    possible = list(getattr(results, "label_possible", ()) or ())
    if declares:
        cards.append(("red", "Label declares: " + nice_list(declares, ALLERGEN_LABEL),
                      "Named in the label's own “Contains …” statement."))
    if possible:
        cards.append(("orange", "Label may declare: " + nice_list(possible, ALLERGEN_LABEL),
                      "Ambiguous wording on the label (for example a bare “nuts”)."))
    for r in results:
        name = r.get("ingredient")
        if r.get("allergen_warning"):
            cards.append(("red", f"{name}: contains " + nice_list(r["allergen_warning"], ALLERGEN_LABEL),
                          "Confirmed allergen hit for your profile."))
        if r.get("possible_allergen_warning"):
            cards.append(("orange", f"{name}: possible " + nice_list(r["possible_allergen_warning"], ALLERGEN_LABEL),
                          "Depends on the source of this ingredient."))
        if r.get("trace_allergen_warning"):
            cards.append(("orange", f"{name}: may contain traces of "
                          + nice_list(r["trace_allergen_warning"], ALLERGEN_LABEL),
                          "Listed under “may contain” or a shared-facility statement."))
        if r.get("condition_warning"):
            cards.append(("yellow", f"{name}: flagged for " + nice_list(r["condition_warning"], CONDITION_LABEL),
                          r.get("notes") or "Condition tag in the knowledge base."))
    return cards


def verdict(results, score, n_items):
    """(key, title, subtitle) derived ONLY from the warnings and the score."""
    declares = list(getattr(results, "label_declares", ()) or ())
    if n_items == 0 and not declares:
        return ("neutral", "Couldn't read ingredients",
                "No ingredients were found. Try a closer photo of the ingredient text, or paste it.")
    cards = collect_warnings(results)
    confirmed = any(level == "red" for level, _, _ in cards)
    if confirmed:
        return ("avoid", "⛔ Avoid", "A confirmed allergen from your profile was found.")
    if score < AVOID_BELOW:
        return ("avoid", "⛔ Avoid", "The health score is very low for this label.")
    if cards:
        return ("caution", "⚠️ Use caution", "Possible or trace allergens or condition flags for your profile.")
    if score < CAUTION_BELOW:
        return ("caution", "⚠️ Use caution", "No warnings for your profile, but the health score is moderate to low.")
    return ("ok", "✅ Looks fine for you", "No warnings for your selected profile and a good health score.")


def score_breakdown(results):
    """Mirror of the documented formula; used only if it reproduces compute_health_score exactly."""
    w = {"low": 0, "medium": 5, "high": 15, "unknown": 3}
    cap = getattr(pipeline, "UNKNOWN_PENALTY_CAP", None)
    rows = [("Start", 100)]
    declares = len(getattr(results, "label_declares", ()) or ())
    possible = len(getattr(results, "label_possible", ()) or ())
    med = high = unk = conf = soft = 0
    conds = set()
    for r in results:
        if r.get("scored", not r.get("trace")):
            lvl = r.get("risk_level")
            if lvl == "medium":
                med += 1
            elif lvl == "high":
                high += 1
            elif lvl == "low":
                pass
            else:
                unk += 1
        conf += len(r.get("allergen_warning") or [])
        soft += len(r.get("possible_allergen_warning") or []) + len(r.get("trace_allergen_warning") or [])
        conds |= set(r.get("condition_warning") or [])
    unk_pts = unk * w["unknown"]
    if cap is not None:
        unk_pts = min(unk_pts, cap)
    rows += [(f"Medium-risk additives × {med} (−5 each)", -5 * med),
             (f"High-risk additives × {high} (−15 each)", -15 * high),
             (f"Unknown items × {unk} (−3 each, capped at 15 in total)", -unk_pts),
             (f"Confirmed allergens × {declares + conf} (−20 each)", -20 * (declares + conf)),
             (f"Possible / trace allergens × {possible + soft} (−10 each)", -10 * (possible + soft)),
             (f"Conditions hit × {len(conds)} (−15 each, counted once)", -15 * len(conds))]
    total = max(0, min(100, sum(p for _, p in rows)))
    return rows if total == compute_health_score(results) else None


def risk_badge(r):
    if r.get("unrecognised"):
        return '<span class="ns-badge b-junk">not scored</span>'
    lvl = str(r.get("risk_level") or "unknown").lower()
    if lvl not in ("low", "medium", "high"):
        lvl = "unknown"
    out = f'<span class="ns-badge b-{lvl}">{lvl}</span>'
    if r.get("trace"):
        out += ' <span class="ns-badge b-trace">may contain</span>'
    return out


SOURCE_LABEL = {"knowledge_base": "Knowledge base", "knowledge_base_fuzzy": "KB (fuzzy match, check)",
                "ml_prediction": "ML guess", "unrecognised": "Unrecognised (OCR noise?)",
                "unmatched": "Not found"}


def ingredients_table_html(results):
    head = "".join(f"<th>{h}</th>" for h in ("Ingredient", "Category", "INS", "Risk", "Matched by"))
    body = ""
    for r in results:
        cat = r.get("category") or ""
        body += ("<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            esc(r.get("ingredient")), esc(str(cat).replace("_", " ")), esc(r.get("ins_code") or "–"),
            risk_badge(r), esc(SOURCE_LABEL.get(r.get("source"), r.get("source") or ""))))
    return f'<div class="ns-scroll"><table class="ns-table"><tr>{head}</tr>{body}</table></div>'


def results_dataframe(results):
    def j(v):
        return ", ".join(v) if v else ""
    return pd.DataFrame([{
        "ingredient": r.get("ingredient"), "category": r.get("category"), "ins_code": r.get("ins_code"),
        "risk_level": r.get("risk_level"), "source": r.get("source"), "may_contain": r.get("trace"),
        "not_scored_ocr_noise": r.get("unrecognised"), "allergen_warning": j(r.get("allergen_warning")),
        "possible_allergen_warning": j(r.get("possible_allergen_warning")),
        "trace_allergen_warning": j(r.get("trace_allergen_warning")),
        "condition_warning": j(r.get("condition_warning")), "notes": r.get("notes")} for r in results])


def report_text(results, score, vtitle, profile, source_name):
    lines = ["NutriScan result", f"Source: {source_name}",
             f"Profile: allergies = {', '.join(profile['allergies']) or 'none selected'}; "
             f"conditions = {', '.join(profile['conditions']) or 'none selected'}",
             f"Verdict: {vtitle}", f"Health score: {score}/100", "", "Warnings:"]
    cards = collect_warnings(results)
    lines += [f"- {t}" for _, t, _ in cards] or ["- none for the selected profile"]
    lines += ["", "Ingredients:"]
    lines += [f"- {r.get('ingredient')} | {r.get('category')} | risk {r.get('risk_level')} | {r.get('source')}"
              for r in results]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading knowledge base and models…")
def load_resources():
    kb = load_knowledge_base()
    clf = vec = None
    try:
        clf = joblib.load(os.path.join(BASE_DIR, "models", "ingredient_classifier.pkl"))
        vec = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
    except Exception:
        clf = vec = None
    return kb, clf, vec


def _load_sample(key):
    st.session_state["mode"] = "Paste text"
    st.session_state["label_text"] = SAMPLES[key]["text"]
    st.session_state["analysed_text"] = SAMPLES[key]["text"]
    st.session_state["source_name"] = SAMPLES[key]["title"]


# ----------------------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="NutriScan", page_icon="🥗", layout="centered",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    kb, clf, vec = load_resources()

    # sidebar: health profile
    with st.sidebar:
        st.markdown("### 🩺 Your health profile")
        allergies = st.multiselect("Allergies", [k for k, _ in ALLERGENS],
                                   format_func=lambda k: ALLERGEN_LABEL[k], key="allergies")
        conditions = st.multiselect("Conditions", [k for k, _ in CONDITIONS],
                                    format_func=lambda k: CONDITION_LABEL[k], key="conditions")
        st.caption("Warnings and the score use only what you select here.")
        if clf is None:
            st.caption("ML fallback model not found: unknown items are shown as not in the knowledge base.")
    profile = {"allergies": allergies, "conditions": conditions}

    # hero
    st.markdown(
        '<div class="ns-hero"><h1>🥗 NutriScan</h1>'
        "<p>Scan an Indian food label. Get allergen and health warnings made for you.</p>"
        '<div class="ns-chips"><span>📷 Label OCR</span><span>🧪 FSSAI / INS knowledge base</span>'
        "<span>⚠️ Personalised warnings</span></div></div>", unsafe_allow_html=True)
    st.caption("Set your allergies and conditions in the sidebar (☰ at the top left on a phone).")

    # input
    st.markdown("**Try a sample label** (typed demo text, not a real photo)")
    keys = list(SAMPLES)
    for row in (keys[:2], keys[2:]):
        cols = st.columns(2)
        for col, k in zip(cols, row):
            col.button(SAMPLES[k]["button"], key="sample_" + k, on_click=_load_sample, args=(k,))

    mode = st.radio("Input", ["Upload photo", "Paste text"], horizontal=True, key="mode",
                    label_visibility="collapsed")
    if mode == "Upload photo":
        up = st.file_uploader("Photo of the ingredient list", type=["png", "jpg", "jpeg", "webp"])
        st.caption("Tip: crop or zoom onto the ingredient text. Whole-pack photos give noisy OCR.")
        if up is not None:
            st.image(up, caption="Your photo")
            if st.button("🔍 Scan label", key="scan"):
                try:
                    suffix = os.path.splitext(up.name)[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(up.getvalue())
                        path = tmp.name
                    with st.spinner("Reading the label (first run loads the OCR model)…"):
                        text = extract_text_from_image(path)
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    st.session_state["analysed_text"] = text
                    st.session_state["source_name"] = "Photo: " + up.name
                except Exception:
                    st.error("We couldn't read this photo. Try a sharper, closer shot of just the "
                             "ingredient text, or switch to “Paste text”.")
    else:
        st.text_area("Ingredient text", key="label_text", height=170,
                     placeholder="Paste the ingredients here, e.g. Ingredients: sugar, milk solids, "
                                 "emulsifier (INS 322(i)) … Contains milk.")
        if st.button("🔍 Analyse text", key="analyse"):
            st.session_state["analysed_text"] = st.session_state.get("label_text", "")
            st.session_state["source_name"] = "Pasted text"

    # results
    text = st.session_state.get("analysed_text", "")
    if not text.strip():
        st.info("Pick a sample above, upload a photo, or paste ingredient text to see results.")
        st.markdown(f'<div class="ns-foot">{esc(DISCLAIMER)}</div>', unsafe_allow_html=True)
        return

    items = clean_ingredient_text(text)
    results = check_warnings(items, kb, clf, vec, profile)
    score = compute_health_score(results)
    vkey, vtitle, vsub = verdict(results, score, len(items))
    source_name = st.session_state.get("source_name", "")

    st.markdown("---")
    if vkey == "neutral":
        bar = ""
    else:
        bar = (f'<div class="ns-scorewrap"><div class="ns-score" style="color:{score_colour(score)}">'
               f"{score}<small> / 100</small></div>"
               f'<div class="ns-bar"><div class="ns-fill" style="width:{score}%;'
               f'background:{score_colour(score)}"></div></div></div>')
    st.markdown(f'<div class="ns-verdict ns-{vkey}"><p class="big">{esc(vtitle)}</p>'
                f'<p class="sub">{esc(vsub)}</p>{bar}</div>', unsafe_allow_html=True)
    if not (allergies or conditions):
        st.caption("No allergies or conditions selected: only additive risk affects the score.")

    tab_w, tab_i, tab_s = st.tabs(["⚠️ Warnings", "🧾 Ingredients", "🧮 How it was scored"])

    with tab_w:
        cards = collect_warnings(results)
        if cards:
            icons = {"red": "🛑", "orange": "🟠", "yellow": "🟡"}
            for level, title, detail in cards:
                st.markdown(f'<div class="ns-card ns-{level}">{icons[level]} <b>{esc(title)}</b>'
                            f'<span class="why">{esc(detail)}</span></div>', unsafe_allow_html=True)
        elif allergies or conditions:
            st.markdown('<div class="ns-card ns-green">✅ <b>No warnings for your selected profile</b></div>',
                        unsafe_allow_html=True)
        else:
            st.info("Select allergies or conditions in the sidebar to get personalised warnings.")
        st.caption("🛑 confirmed allergen · 🟠 possible / trace allergen · 🟡 condition flag")

    with tab_i:
        if len(results):
            st.markdown(ingredients_table_html(results), unsafe_allow_html=True)
            st.caption("“ML guess” = not in the knowledge base; only a category is predicted and the "
                       "risk stays unknown. “Not scored” = looks like OCR noise.")
        else:
            st.info("No ingredient rows to show.")
        with st.expander("Text used for the analysis"):
            st.text(text)

    with tab_s:
        st.markdown(
            "**Health score** starts at 100 and subtracts:\n"
            "- 5 per medium-risk additive, 15 per high-risk additive\n"
            "- 3 per unknown item, **capped at 15 in total**\n"
            "- 20 per confirmed allergen hit (only allergies you selected)\n"
            "- 10 per possible or trace allergen hit (only allergies you selected)\n"
            "- 15 per selected condition that is hit (counted once)\n\n"
            "The result is clamped to 0–100. Junk OCR items are listed but not scored.")
        rows = score_breakdown(results)
        if rows:
            st.markdown("**This label:**")
            st.table(pd.DataFrame([(a, b) for a, b in rows] + [("Final score", score)],
                                  columns=["Step", "Points"]).set_index("Step"))
        st.caption(f"Verdict bands (UI design choice, ASSUMED): Avoid if a confirmed allergen is found "
                   f"or score < {AVOID_BELOW}; Use caution if any other warning or score < {CAUTION_BELOW}; "
                   "otherwise Looks fine. The score is a team-designed heuristic, not medical advice.")

    # downloads
    st.markdown("&nbsp;", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Results (CSV)", results_dataframe(results).to_csv(index=False).encode("utf-8-sig"),
                       file_name="nutriscan_results.csv", mime="text/csv")
    c2.download_button("⬇️ Summary (text)", report_text(results, score, vtitle, profile, source_name),
                       file_name="nutriscan_summary.txt", mime="text/plain")

    st.markdown(f'<div class="ns-foot">{esc(DISCLAIMER)}</div>', unsafe_allow_html=True)


main()
