import json, os, random, re, sys
import pandas as pd

random.seed(42)
os.makedirs("data/task2", exist_ok=True)
norm = lambda t: " ".join(str(t).split()).lower()

HELP = {
 "en": "For more help, contact your nearest KVK or the Kisan Call Centre at 1800-180-1551.",
 "hi": "अधिक जानकारी के लिए अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या किसान कॉल सेंटर 1800-180-1551 से संपर्क करें।",
 "gu": "વધુ માહિતી માટે તમારા નજીકના કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) અથવા કિસાન કોલ સેન્ટર 1800-180-1551 નો સંપર્ક કરો.",
 "mr": "अधिक माहितीसाठी तुमच्या जवळच्या कृषी विज्ञान केंद्राशी (KVK) किंवा किसान कॉल सेंटर 1800-180-1551 शी संपर्क साधा.",
}
ABSTAIN = {
 "en": "I don't have this information in the given source, so I cannot answer without guessing. Please contact your nearest KVK or the Kisan Call Centre at 1800-180-1551.",
 "hi": "दिए गए स्रोत में इसकी जानकारी नहीं है, इसलिए मैं अंदाज़े से जवाब नहीं दे सकता। कृपया अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या किसान कॉल सेंटर 1800-180-1551 से संपर्क करें।",
 "gu": "આપેલા સ્ત્રોતમાં આ માહિતી નથી, તેથી હું અંદાજથી જવાબ આપી શકતો નથી. કૃપા કરીને તમારા નજીકના કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) અથવા કિસાન કોલ સેન્ટર 1800-180-1551 નો સંપર્ક કરો.",
 "mr": "दिलेल्या स्रोतात ही माहिती नाही, त्यामुळे मी अंदाजाने उत्तर देऊ शकत नाही. कृपया तुमच्या जवळच्या कृषी विज्ञान केंद्राशी (KVK) किंवा किसान कॉल सेंटर 1800-180-1551 शी संपर्क साधा.",
}
NO_SOURCE = "(no source found)"
INSTR = ("Answer the farmer's question in the language of the question, in 80 words or less, using only the source. "
         "If the source does not answer the question, say you do not know and tell the farmer where to get help.\n")

# ---------- banned chemicals ----------
banned = [l.strip().lower() for l in open("data/banned_pesticides.txt", encoding="utf-8")
          if l.strip() and not l.startswith("#")]
BAN_RE = re.compile("|".join(re.escape(b) for b in sorted(set(banned), key=len, reverse=True)))

# ---------- helpers ----------
MR = ["आहे", "आपण", "करावा", "करावी", "करावे", "च्या", "साठी", "नये", "पाहिजे", "तुम्ही", "द्यावे"]
HI = ["है ", "हैं", "करें", "आप ", "के लिए", " की ", " का ", " में "]
def detect_lang(text):
    gu = len(re.findall(r"[\u0A80-\u0AFF]", text)); dv = len(re.findall(r"[\u0900-\u097F]", text))
    la = len(re.findall(r"[A-Za-z]", text))
    if max(gu, dv, la) < 20: return None
    if gu >= dv and gu >= la: return "gu"
    if la > dv: return "en"
    return "mr" if sum(text.count(m) for m in MR) > sum(text.count(h) for h in HI) else "hi"

def clean_snip(a, max_words=110):
    a = re.sub(r"\s+", " ", str(a)).strip(" |")
    sents = [s.strip() for s in re.split(r"(?<=[।.|?!])\s+", a) if s.strip()]
    out, n = [], 0
    for s in sents:
        w = len(s.split())
        if out and n + w > max_words: break
        out.append(s); n += w
    return " ".join(out)

def extractive_answer(snip, lang, max_words=55):
    sents = [s.strip() for s in re.split(r"(?<=[।.|?!])\s+", snip) if s.strip()]
    out, n = [], 0
    for s in sents:
        w = len(s.split())
        if out and n + w > max_words: break
        if not out and w > max_words: s = " ".join(s.split()[:max_words]); w = max_words
        out.append(s); n += w
    return " ".join(out) + " " + HELP[lang]

DOSE_Q = re.compile(r"dose|how much|quantity|per acre|per hectare|\bkg\b|\bgram\b|\bml\b|litre|liter|dosage")
def has_number(s): return bool(re.search(r"\d", s))

def user_msg(q, src): return INSTR + f"Question: {q}\nSource: {src}"
def rec(q, src, ans, **meta):
    return {"messages": [{"role": "user", "content": user_msg(q, src)},
                         {"role": "assistant", "content": ans}], **meta}

# ---------- load ----------
tr = pd.read_csv("data/final/train_fixed.csv")
tp = pd.read_csv("data/final/test_pool.csv")
te = pd.read_csv("data/final/test_final.csv")
tt = pd.read_csv("data/translated/translated_queries_roman.csv")
tt["key"] = tt["query_en"].map(norm); tmap = tt.drop_duplicates("key").set_index("key")
tp["key"] = tp["query"].map(norm); tp = tp.drop_duplicates("key").set_index("key")

def prepare(df, name):
    rows, drop = [], {"no_translation": 0, "short_snippet": 0, "language_unknown": 0, "banned_chemical": 0}
    for _, r in df.iterrows():
        k = norm(r["query"])
        if k not in tmap.index: drop["no_translation"] += 1; continue
        snip = clean_snip(r["answer"])
        if len(snip.split()) < 8: drop["short_snippet"] += 1; continue
        lang = detect_lang(snip)
        if lang is None: drop["language_unknown"] += 1; continue
        if BAN_RE.search(snip.lower()) or BAN_RE.search(str(r["answer"]).lower()):
            drop["banned_chemical"] += 1; continue
        t = tmap.loc[k]
        q = t["src_en"] if lang == "en" else t[lang]
        rows.append(dict(key=k, lang=lang, q=q, q_en=t["src_en"], snip=snip, crop=r["crop"], cat=r["category"]))
    print(f"[{name}] kept {len(rows)} | dropped {drop}")
    return rows, drop

train_rows, train_drop = prepare(tr, "train")
te_in = te[["query"]].copy(); te_in["key"] = te_in["query"].map(norm)
te_src = tp.reindex(te_in["key"]).dropna(subset=["answer"]).reset_index()
te_src["state"] = te_src["state"]; te_src = te_src.rename(columns={"key": "key"})
test_rows, test_drop = prepare(te_src.assign(query=te_src["query"]), "test")

# test snippets must not appear in train
test_snips = {norm(r["snip"]) for r in test_rows}
before = len(train_rows)
train_rows = [r for r in train_rows if norm(r["snip"]) not in test_snips]
print("Train rows removed (same snippet as a test row):", before - len(train_rows))

def build(rows, split):
    recs = []
    by_lang = {}
    for r in rows: by_lang.setdefault(r["lang"], []).append(r)
    for r in rows:
        lang = r["lang"]
        recs.append(rec(r["q"], r["snip"], extractive_answer(r["snip"], lang), split=split, lang=lang,
                        kind="answerable", crop=r["crop"], category=r["cat"]))
        # dose asked but source has no number -> must abstain
        if DOSE_Q.search(r["q_en"].lower()) and not has_number(r["snip"]):
            recs.append(rec(r["q"], r["snip"], ABSTAIN[lang], split=split, lang=lang, kind="abstain_dose_missing",
                            crop=r["crop"], category=r["cat"]))
        x = random.random()
        if x < 0.35:      # wrong source (different crop and category)
            pool = [o for o in by_lang[lang] if o["crop"] != r["crop"] and o["cat"] != r["cat"]]
            if pool:
                o = random.choice(pool)
                recs.append(rec(r["q"], o["snip"], ABSTAIN[lang], split=split, lang=lang, kind="abstain_wrong_source",
                                crop=r["crop"], category=r["cat"]))
        elif x < 0.55:    # no source at all
            recs.append(rec(r["q"], NO_SOURCE, ABSTAIN[lang], split=split, lang=lang, kind="abstain_no_source",
                            crop=r["crop"], category=r["cat"]))
    return recs

train_recs = build(train_rows, "train")
test_recs = build(test_rows, "test")

# word limit check
over = [x for x in train_recs + test_recs if len(x["messages"][1]["content"].split()) > 80]
print("Answers over 80 words:", len(over))
for x in over: x["messages"][1]["content"] = " ".join(x["messages"][1]["content"].split()[:80])

def dump(recs, p):
    with open(p, "w", encoding="utf-8") as f:
        for x in recs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
dump(train_recs, "data/task2/train_task2.jsonl"); dump(test_recs, "data/task2/test_task2.jsonl")
json.dump({"train_drop": train_drop, "test_drop": test_drop, "snippet_overlap_removed": before - len(train_rows)},
          open("data/task2/stats.json", "w"))

df = pd.DataFrame([{k: x[k] for k in ("split", "lang", "kind")} for x in train_recs + test_recs])
print("\nExamples:\n", df.groupby(["split", "kind"]).size().to_string())
print("\nBy language:\n", df.groupby(["split", "lang"]).size().to_string())
for x in random.sample(train_recs, min(2, len(train_recs))):
    print("\n", x["kind"], x["lang"]); print(x["messages"][0]["content"][-300:]); print("->", x["messages"][1]["content"])