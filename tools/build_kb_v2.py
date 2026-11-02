"""Part A: build nutriscan_kb_v2.csv, risk_rules.csv, rows_to_verify.csv, training_variants_v2.csv.
Originals are read-only inputs and are never modified."""
import re
import pandas as pd

UP = "/mnt/user-data/uploads/"
kb = pd.read_csv(UP + "nutriscan_kb_final.csv")
tv = pd.read_csv(UP + "nutriscan_training_variants.csv")
n_before = len(kb)
print("rows before:", n_before)
print("category counts BEFORE:\n", kb.category.value_counts().to_string())
print("risk counts BEFORE:\n", kb.risk_level.value_counts().to_string())

def drop_seg(notes, pattern):
    if pd.isna(notes):
        return ""
    segs = [s for s in str(notes).split(" | ") if not re.search(pattern, s)]
    return " | ".join(segs)

def add_note(notes, text):
    notes = "" if pd.isna(notes) else str(notes)
    return (notes + " | " + text) if notes else text

# ---------------- 1. SOY / possible_allergen ----------------
kb["possible_allergen"] = "no"
m = kb.ingredient_name == "lecithin"
assert m.sum() == 1 and kb.loc[m, "ins_code"].iloc[0] == "INS322"
kb.loc[m, "allergen_tags"] = "soy"
# the old "soy not tagged ..." note is now false -> replaced by the requested wording
kb.loc[m, "notes"] = kb.loc[m, "notes"].apply(lambda n: drop_seg(n, r"^soy not tagged"))
kb.loc[m, "notes"] = kb.loc[m, "notes"].apply(lambda n: add_note(
    n, "INS 322 lecithin is often soy-derived; label does not always say; warn as possible"))
kb.loc[m, "possible_allergen"] = "yes"

SRC_DEPENDENT = {  # ingredient_name -> why
    "mono and diglycerides of fatty acids": "INS 471",
    "diacetyltartric and fatty acid ester of glycerol": "INS 472e",
    "polysorbate 20": "INS 432 (430-series)",
    "polysorbate 80": "INS 433 (430-series)",
    "natural flavour": "natural flavour",
}
for name, why in SRC_DEPENDENT.items():
    mm = kb.ingredient_name == name
    assert mm.sum() == 1, name
    kb.loc[mm, "possible_allergen"] = "yes"
    kb.loc[mm, "notes"] = kb.loc[mm, "notes"].apply(lambda n: add_note(
        n, f"possible allergen: {why} allergen depends on source and label may not say; "
           "no allergen tag assigned (team)"))

# ---------------- 2. OTHER class re-class ----------------
kb["category_v1"] = kb["category"]
ANIMAL = {"egg", "egg powder", "fish", "prawn"}
SPICE = {"spices", "chilli powder", "onion powder", "garlic powder", "turmeric powder",
         "tomato powder", "black pepper", "natural flavour", "added flavour",
         "nature identical flavouring"}
other_mask = kb.category_v1 == "other"
kb.loc[kb.category_v1 == "dairy", "category"] = "animal_derived"
kb.loc[other_mask & kb.ingredient_name.isin(ANIMAL), "category"] = "animal_derived"
kb.loc[other_mask & kb.ingredient_name.isin(SPICE), "category"] = "spice_flavouring"
assert kb.loc[other_mask & kb.ingredient_name.isin(ANIMAL | SPICE)].shape[0] == len(ANIMAL | SPICE)
cc = kb.category.value_counts()
small = cc[cc < 5].index.tolist()
if small:
    kb.loc[kb.category.isin(small), "category"] = "other"
print("\ncategory counts AFTER:\n", kb.category.value_counts().to_string())
print("classes merged into other because <5 rows:", small or "none")

# ---------------- 3. RISK LEVELS ----------------
def ins_base(code):
    if pd.isna(code):
        return None
    mm = re.search(r"(\d+)", str(code))
    return int(mm.group(1)) if mm else None

HIGH = {102: "synthetic azo colour", 110: "synthetic azo colour", 122: "synthetic azo colour",
        124: "synthetic azo colour", 129: "synthetic azo colour",
        320: "synthetic antioxidant BHA", 321: "synthetic antioxidant BHT",
        211: "synthetic preservative sodium benzoate"}
MED_PRES = {200, 202, 210, 212, 280, 281, 282, 310, 319, 249, 250, 251, 252}
MED_SWEET = {950, 951, 952, 954, 955}

def classify(row):
    b = ins_base(row.ins_code)
    if b in HIGH:
        return "high", f"team rule: {HIGH[b]}"
    if b is not None and 220 <= b <= 228:
        return "medium", "team rule: sulphite preservative"
    if b in (249, 250, 251, 252):
        return "medium", "team rule: nitrite/nitrate"
    if b in MED_SWEET:
        return "medium", "team rule: artificial sweetener"
    if row.category_v1 == "flavour_enhancer":
        return "medium", "team rule: flavour enhancer"
    if b in MED_PRES:
        return "medium", "team rule: other synthetic preservative/antioxidant"
    return "low", ("team rule: plain food / ingredient, no listed risk class (default low)"
                   if b is None else "team rule: additive not in a listed risk class (default low)")

kb["risk_level_v1"] = kb["risk_level"]
res = kb.apply(classify, axis=1, result_type="expand")
kb["risk_level"], kb["risk_reason"] = res[0], res[1]
# old notes carried the previous risk wording -> would now contradict risk_level
kb["notes"] = kb["notes"].apply(lambda n: drop_seg(n, r"^(risk (low|medium|high)\b|low: plain food)"))
chg = kb[kb.risk_level != kb.risk_level_v1][["ingredient_name", "ins_code", "risk_level_v1", "risk_level", "risk_reason"]]
chg.to_csv("data/risk_changes.csv", index=False)
print("\nrisk counts AFTER:\n", kb.risk_level.value_counts().to_string())
print(f"\nrisk changed on {len(chg)} rows:\n", chg.to_string(index=False))

rules = pd.DataFrame([
    ["high", "synthetic azo / artificial colours", "INS 102, 110, 122, 124, 129", "team rule"],
    ["high", "synthetic antioxidants BHA, BHT", "INS 320, 321", "team rule"],
    ["high", "sodium benzoate", "INS 211", "team rule"],
    ["medium", "sulphites", "INS 220-228", "team rule"],
    ["medium", "nitrites / nitrates", "INS 249-252 (none in current KB)", "team rule"],
    ["medium", "artificial sweeteners", "INS 950, 951, 952, 954, 955", "team rule"],
    ["medium", "monosodium glutamate and other flavour enhancers", "INS 621, 627, 631, 635, 637", "team rule"],
    ["medium", "other synthetic preservatives / antioxidant-preservatives",
     "INS 200, 202, 210, 212, 280, 281, 282, 310, 319", "team rule"],
    ["low", "everything else, incl. plain foods (milk, peanut, wheat flour)", "all other rows", "team rule"],
    ["note", "allergen status never changes risk_level (handled per user in warnings)", "-", "team rule"],
], columns=["risk_level", "additive_class", "ins_codes", "basis"])
rules.to_csv("data/risk_rules.csv", index=False)

# ---------------- 4. SOURCES ----------------
is_ns = (kb.verification_status == "needs_review") & kb.source.str.contains("FSSAI", na=False)
kb.loc[is_ns, "source"] = kb.loc[is_ns, "source"] + " | tags assigned by team"
print("\nsource suffix appended on", int(is_ns.sum()), "rows")
pool = kb[(kb.verification_status == "sourced") | kb.source.str.contains(r"\bpp?\.\s*\d", na=False)]
print("verification pool size:", len(pool))
pool.sample(12, random_state=42)[["ingredient_name", "ins_code", "source"]].to_csv(
    "data/rows_to_verify.csv", index=False)

# ---------------- save KB v2 ----------------
orig_cols = list(pd.read_csv(UP + "nutriscan_kb_final.csv", nrows=0).columns)
kb_out = kb[orig_cols + ["risk_reason", "category_v1", "possible_allergen"]]
kb_out.to_csv("data/nutriscan_kb_v2.csv", index=False)
chk = pd.read_csv(UP + "nutriscan_kb_final.csv")
assert len(kb_out) == n_before == len(chk)
chg_cols = [c for c in orig_cols if c not in ("category", "risk_level", "allergen_tags", "notes", "source")]
assert (kb_out[chg_cols].fillna("").values == chk[chg_cols].fillna("").values).all()

# ---------------- 5. training variants v2 ----------------
cat_map = dict(zip(kb.ingredient_name, kb.category))
tv2 = tv.copy()
tv2["category"] = tv2["parent_ingredient"].map(cat_map)
assert tv2.category.notna().all()
tv2.to_csv("data/nutriscan_training_variants_v2.csv", index=False)
print("\ntraining rows:", len(tv2), "\n", tv2.category.value_counts().to_string())
print("rows whose category changed:", int((tv2.category != tv.category).sum()))
