# NutriScan: AI Food Label Decoder for Indian Packaged Food

NutriScan is a Streamlit web app that reads the ingredient list on a packaged-food label (from a photo or
pasted text) and gives personalised allergen and health warnings. The user picks their allergies and health
conditions, and the app flags the ingredients that matter for them and shows a simple health score.

Final mini-project for **ERD481 (ML minor)**.
Team: Bhagyasree Kishore Menon (LBT23ECE016) and Diya Sony (LBT23ECE024).
Guide: Prof. Gisha G S, Dept. of CSE.

> **Not medical advice.** Allergen, condition and risk tags are rules assigned by the team and still need
> expert review. Always read the printed label.

---

## What it does

1. **Profile:** the user selects allergies (dairy, gluten, soy, peanut, tree nut, egg, shellfish, fish,
   sulphites, phenylketonuria) and conditions (diabetes, hypertension, asthma).
2. **Input:** a photo of the ingredient list, or pasted ingredient text.
3. **Output:** warnings (confirmed allergens, possible or trace allergens, condition risks), a table of the
   ingredients found with category, INS code and risk level, and a health score from 0 to 100.

It is a **web app** that opens in a phone or laptop browser. It is not a native mobile app.

## How it works

```
Photo or text
   -> EasyOCR (photo is rotated and shrunk first)
   -> Regex cleaning (removes nutrition-table rows; reads "Contains ..." and "May contain ..." statements,
      bracket sub-ingredients and INS codes; repairs common OCR mistakes)
   -> Knowledge-base lookup (exact name, alias, INS code, fuzzy match)
   -> TF-IDF + SVM classifier: predicts a category ONLY for ingredients not found in the knowledge base
   -> Rule-based allergen and condition warnings (plus an allergen keyword safety net)
   -> Health score
```

**Health score:** starts at 100. Minus 5 per medium-risk and 15 per high-risk additive; minus 3 per "unknown"
item (at most 15 in total); minus 20 per confirmed allergen; minus 10 per possible or trace allergen; minus 15
per selected condition that is hit (counted once). Only the allergies and conditions the user selected count.
The result is limited to 0 to 100. The risk rules are in `docs/risk_rules.csv`.

## Data

- `data/nutriscan_kb_v2.csv`: knowledge base of **189 ingredients** (186 base rows plus 3 INS codes added
  after a real-label test). It combines additives extracted from FSSAI documents with common label
  ingredients. **Allergen, condition and risk tags were assigned by the team as rules. They are not FSSAI
  data, and most rows are marked `needs_review` in `verification_status`.**
- `data/nutriscan_training_variants_v2.csv`: ingredient names, aliases and OCR-style spelling variants
  (1,058 rows), used to train the classifier. The 3 later INS rows were not used in training.
- Ingredients are grouped into **13 categories**.

## Classifier results

Five models were compared with 5-fold GroupKFold, grouped by parent ingredient, so every model is tested on
ingredients it has not seen (`models/model_comparison.csv`):

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| LinearSVC (SVM) | 0.654 | 0.654 |
| Random Forest | 0.615 | 0.623 |
| KNN (k=3) | 0.600 | 0.608 |
| Logistic Regression | 0.578 | 0.568 |
| Decision Tree | 0.502 | 0.527 |

LinearSVC had the highest accuracy and F1 and is the model the app uses. Per-class scores are in
`models/classification_report.txt` and the confusion matrix is in `models/confusion_matrix.png`.
The scores are moderate because the test uses unseen ingredients and some classes are small or mixed
(for example `other`). The classifier is only a fallback; most warnings come from the knowledge base and rules.

## Run it

Requirements: Python with a virtual environment. Developed and tested on Windows. The first photo upload
downloads EasyOCR's language models, so you need internet for that first run.

```bash
# 1. create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 2. install packages
pip install -r requirements.txt

# 3. start the app (run from the project folder)
streamlit run app.py
```

Open `http://localhost:8501`. To use it from a phone, connect the phone to the same Wi-Fi, open the
**Network URL** that Streamlit prints in the terminal, and use the photo upload (choose "Take photo").

Tips for photos: zoom in so the ingredient text fills the frame, avoid glare, and leave out the nutrition
table. If a photo cannot be read, use **Paste ingredient text**.

**Retrain the classifier (optional, it overwrites the saved model):** `python train_classifier.py`

**Run the tests:** `python test_pipeline.py` (8 typed label texts plus a real OCR text; it should end
with `All assertions passed.`)

## Project structure

```
NutriScan/
├── app.py                  Streamlit interface
├── pipeline.py             OCR, cleaning, matching, warnings, health score
├── train_classifier.py     trains and compares the five models
├── test_pipeline.py        label-text tests
├── requirements.txt
├── data/                   knowledge base and training CSVs
├── models/                 saved classifier and vectorizer, model comparison, classification report,
│                           confusion matrix
├── docs/                   risk rules and test output
└── tools/                  build_kb_v2.py (script that built the knowledge base)
```

## Real-label testing so far

| Product | Photo | Result |
|---|---|---|
| Peanut chocolate bar (screenshot) | Clear | Worked; declared peanut and soy for a soy + peanut user |
| Veeba veg mayonnaise (jar) | Uncropped | Noisy OCR |
| Veeba veg mayonnaise (jar) | Zoomed on ingredients | Nearly all ingredients read; dairy declared |
| Instant noodles | Whole panel | Allergen declarations correct; OCR noisy |

## Limitations

- Web app only; no native mobile app.
- OCR depends on photo quality. Curved jars and whole-panel photos are noisy; zooming on the ingredient
  text helps.
- Small knowledge base; common ingredients that are missing fall back to the classifier's guess.
- Allergen, condition and risk tags are team-assigned and need expert review.
- Limited real-label testing and no OCR accuracy figure yet.
- The health score is a team-designed heuristic, not a medical measure.
- The allergen keyword check is deliberately conservative; for example, it flags soy-derived oil even when
  the label's allergen statement lists only milk.

## Future work

A larger, expert-reviewed knowledge base; an in-app crop tool; OCR accuracy testing on many labels;
retraining the classifier on the extended knowledge base; a native mobile app.

## References

1. E. Kına, "Real-time food allergen detection using OCR-enhanced machine learning techniques," *PeerJ
   Computer Science*, vol. 11, p. e3338, 2025. https://doi.org/10.7717/peerj-cs.3338
2. B. K. Rai, N. S. Chandan, D. N. Marangappanavar, S. Indira, and G. Kumar, "Classifying food ingredients
   using machine learning on nutritional and biochemical data," *Discover Food*, vol. 5, 2025.
   https://doi.org/10.1007/s44187-025-00661-7
3. A. Rawat, K. Joshi, A. Joshi, M. Diwakar, et al., "Food label analysis for ingredient extraction and
   prediction," in *2025 IEEE 7th Int. Conf. Computing, Communication and Automation (ICCCA)*, Nov. 2025.
   https://doi.org/10.1109/ICCCA66364.2025.11325614
