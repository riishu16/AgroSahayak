import json, random, re, sys, os
import pandas as pd
sys.path.insert(0, "scripts")
from make_roman import to_roman, add_noise
from indic_transliteration import sanscript

random.seed(42)
os.makedirs("data/task1", exist_ok=True)

def norm(t):
    return " ".join(str(t).split()).lower()

# ---------- 1. entity (keyword list se) ----------
ENT = ["fall armyworm","armyworm","fruit fly","fruit borer","stem borer","shoot borer","pod borer",
 "heliothis","helicoverpa","white grub","termite","thrips","whitefly","mealybug","leaf miner",
 "cutworm","weevil","jassid","hopper","caterpillar","nematode","aphid","mite",
 "panama wilt","early blight","late blight","leaf blight","powdery mildew","downy mildew",
 "mildew","blight","wilt","rust","blast","smut","fruit rot","root rot","rhizome rot","rot",
 "leaf curl","mosaic","sigatoka","anthracnose","damping off","gummosis","fungus","fungal",
 "nano urea","nano dap","urea","dap","npk","zinc sulphate","zinc","boron","sulphur",
 "potash","calcium","magnesium","humic acid","micronutrient","vermicompost","gypsum",
 "weedicide","herbicide","insecticide","fungicide","pesticide","weed","drought"]
ENT_NORM = {"aphids":"aphid","fungal":"fungus","zink":"zinc","armyworm":"fall armyworm"}
ENT_RE = re.compile(r"\b(" + "|".join(sorted(ENT, key=len, reverse=True)) + r")s?\b")
def entity_of(q):
    m = ENT_RE.search(q.lower())
    if not m: return "none"
    e = m.group(1)
    return ENT_NORM.get(e, e)

# ---------- 2. crop: sawaal me crop ka naam nahi to unknown ----------
ALIAS = {"chickpea":["chickpea","chick pea","gram","chana","bengal"], "pigeon pea":["pigeon","arhar","tur","red gram"],
 "mung bean":["green gram","moong","mung"], "urd bean":["black gram","urd","urad"], "lentil":["lentil","masur","masoor"],
 "pearl millets":["bajra","bajara","pearl millet"], "paddy (dhan)":["paddy","rice","dhan"], "rice":["paddy","rice","dhan"],
 "mustard":["mustard","raya","rai"], "okra":["okra","bhindi","bhendi"], "sesame":["sesame","sesamum","til"],
 "maize":["maize","makka","corn"], "bottle gourd":["bottle","lauki","gaurd","gourd"], "brinjal":["brinjal","baingan","eggplant"],
 "cotton":["cotton"], "sugarcane":["sugarcane","ratoon"]}
def crop_in_text(crop, q):
    ql = q.lower()
    return any(a in ql for a in ALIAS.get(crop, [crop]))

# ---------- 3. synthetic: weather + post_harvest ----------
CROPS = [("wheat","गेहूं"),("maize","मक्का"),("onion","प्याज"),("potato","आलू"),("tomato","टमाटर"),("mustard","सरसों"),("rice","धान")]
W_EN = ["Will it rain this week in my area","What is the weather forecast for the next 5 days","Temperature forecast for tomorrow",
        "Is there any chance of rainfall today","Weather update for my village","Will cold wave affect {c}? weather information"]
W_HI = ["इस हफ्ते मेरे इलाके में बारिश होगी क्या","अगले 5 दिन का मौसम कैसा रहेगा","कल का तापमान कितना रहेगा",
        "आज बारिश की संभावना है क्या","मेरे गांव के मौसम की जानकारी दीजिए"]
P_EN = ["How to store {c} after harvest","Packaging and grading of {c}","Storage method for {c} to keep it fresh",
        "How to increase shelf life of {c}","Cold storage information for {c}","Post harvest handling of {c}","How to dry {c} after harvest before storing"]
P_HI = ["कटाई के बाद {h} का भंडारण कैसे करें","{h} की ग्रेडिंग और पैकिंग कैसे करें","{h} को ज्यादा दिन तक सुरक्षित कैसे रखें","कटाई के बाद {h} की देखभाल कैसे करें"]
def synthetic():
    rows = []
    for t in W_EN:
        for c, h in random.sample(CROPS, 2) if "{c}" in t else [("", "")]:
            q = t.format(c=c)
            rows.append(dict(q=q, lang="en", crop=c if "{c}" in t else "unknown", category="weather"))
    for t in W_HI:
        rows.append(dict(q=t, lang="hi", crop="unknown", category="weather"))
    for t in P_EN:
        for c, h in random.sample(CROPS, 3):
            rows.append(dict(q=t.format(c=c), lang="en", crop=c, category="post_harvest"))
    for t in P_HI:
        for c, h in random.sample(CROPS, 3):
            rows.append(dict(q=t.format(h=h), lang="hi", crop=c, category="post_harvest"))
    return rows

# ---------- 4. prompt ----------
INSTR = ("Extract crop, query_category and specific_entity from this farmer query. "
         "Answer with JSON only.\nQuery: ")
def record(q, crop, cat, ent, **meta):
    gold = {"crop": crop, "query_category": cat, "specific_entity": ent}
    return {"messages": [{"role": "user", "content": INSTR + q},
                         {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)}],
            "gold": gold, **meta}

VARS = [("en","native","src_en"),("hi","native","hi"),("gu","native","gu"),("mr","native","mr"),
        ("hi","roman","hi_roman_noisy"),("gu","roman","gu_roman_noisy"),("mr","roman","mr_roman_noisy")]

tr = pd.read_csv("data/final/train_fixed.csv")
te = pd.read_csv("data/final/test_final.csv")
tt = pd.read_csv("data/translated/translated_queries_roman.csv")
tt["key"] = tt["query_en"].map(norm)
tmap = tt.drop_duplicates("key").set_index("key")

# ---------- 5. train ----------
train_recs, miss, n_unknown = [], 0, 0
for _, r in tr.iterrows():
    k = norm(r["query"])
    if k not in tmap.index: miss += 1; continue
    t = tmap.loc[k]
    q_en = " ".join(str(r["query"]).split())
    crop = r["crop"] if crop_in_text(r["crop"], q_en) else "unknown"
    n_unknown += crop == "unknown"
    ent = entity_of(q_en)
    train_recs.append(record(t["src_en"], crop, r["category"], ent, lang="en", script="native", synthetic=False))
    for lang, script, col in random.sample(VARS[1:], 3):          # har sawaal ke 3 aur roop
        train_recs.append(record(t[col], crop, r["category"], ent, lang=lang, script=script, synthetic=False))
for s in synthetic():
    ent = "none"
    train_recs.append(record(s["q"], s["crop"], s["category"], ent, lang=s["lang"], script="native", synthetic=True))
    if s["lang"] == "hi":
        rom = add_noise(to_roman(s["q"], sanscript.DEVANAGARI))
        train_recs.append(record(rom, s["crop"], s["category"], ent, lang="hi", script="roman", synthetic=True))

# ---------- 6. test (7 roop x 100) ----------
test_recs = []
for _, r in te.iterrows():
    k = norm(r["query"])
    if k not in tmap.index:
        print("WARNING: test sawaal ka translation nahi mila, id", r["id"]); continue
    t = tmap.loc[k]
    for lang, script, col in VARS:
        text = " ".join(str(r["query"]).split()) if col == "src_en" else t[col]
        test_recs.append(record(text, r["crop"], r["category"], r["entity"],
                                id=int(r["id"]), lang=lang, script=script, synthetic=False))

# ---------- 7. contamination check ----------
train_txt = {norm(x["messages"][0]["content"][len(INSTR):]) for x in train_recs}
test_txt = {norm(x["messages"][0]["content"][len(INSTR):]) for x in test_recs}
overlap = train_txt & test_txt
overlap_before = len(overlap)
print("Exact overlap train-test (hatane se pehle):", len(overlap))
before = len(train_recs)
train_recs = [x for x in train_recs
              if norm(x["messages"][0]["content"][len(INSTR):]) not in test_txt]
print("Train examples removed (text test se match hua):", before - len(train_recs))
train_txt = {norm(x["messages"][0]["content"][len(INSTR):]) for x in train_recs}
print("Overlap ab:", len(train_txt & test_txt), "(0 hona chahiye)")

def dump(recs, path):
    with open(path, "w", encoding="utf-8") as f:
        for x in recs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
dump(train_recs, "data/task1/train_task1.jsonl")
dump(test_recs, "data/task1/test_task1.jsonl")

json.dump({"overlap_before": overlap_before, "train_removed": before - len(train_recs),
           "overlap_after": len(train_txt & test_txt), "unknown_crop_rows": int(n_unknown),
           "train_rows_used": int(len(tr)) - miss, "translation_missing": miss},
          open("data/task1/stats.json", "w"))
print("\nTrain examples:", len(train_recs), "| Test examples:", len(test_recs), "| translation nahi mila:", miss)
print("crop=unknown (sawaal me crop nahi):", n_unknown)
tdf = pd.DataFrame([{**x["gold"], "synthetic": x["synthetic"], "lang": x["lang"], "script": x["script"]} for x in train_recs])
print("\nCategory counts (train):\n", tdf["query_category"].value_counts().to_string())
print("\nEntity != none:", (tdf["specific_entity"] != "none").mean().round(2))
print("Synthetic examples:", int(tdf["synthetic"].sum()))
print("\nlang/script (train):\n", tdf.groupby(["lang","script"]).size().to_string())
print("\n3 sample:")
for x in random.sample(train_recs, 3):
    print(x["messages"][0]["content"][len(INSTR):], "->", x["messages"][1]["content"])
