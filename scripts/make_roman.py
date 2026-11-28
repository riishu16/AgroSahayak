import re, random
import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

random.seed(42)

def itrans_to_simple(w):
    # schwa deletion at word end: "prabandhana" -> "prabandhan"
    if len(w) > 2 and w.endswith("a") and w[-2] not in "aeiouAEIOU":
        w = w[:-1]
    for a, b in [(".Dh", "dh"), (".D", "d"), (".N", "n"), ("Chh", "chh"), ("Ch", "chh"),
                 ("chh", "chh"), ("shh", "sh"), ("Sh", "sh"), ("GY", "gy"), ("j~n", "gy"),
                 ("kSh", "ksh"), ("R^i", "ri"), ("~N", "n"), ("~n", "n"),
                 ("Th", "th"), ("Dh", "dh"), ("M", "n"), ("H", "h")]:
        w = w.replace(a, b)
    for a, b in [("A", "a"), ("I", "i"), ("U", "u"), ("E", "e"), ("O", "o"),
                 ("T", "t"), ("D", "d"), ("N", "n"), ("L", "l"), ("S", "sh")]:
        w = w.replace(a, b)
    return w

def to_roman(text, script):
    s = transliterate(str(text), script, sanscript.ITRANS)
    s = " ".join(itrans_to_simple(w) for w in s.split()).lower()
    s = s.replace("\u0949", "o").replace("\u0945", "e").replace("\u0ac9", "o").replace("\u0ac5", "e")
    return re.sub(r"[^\x00-\x7f]", "", s)          # koi bhi bacha hua non-ASCII hatao

def add_noise(text):
    words = text.split()
    out = []
    for w in words:
        r = random.random()
        if len(w) > 4 and r < 0.12:               # ek vowel chhoot gaya
            vs = [i for i, c in enumerate(w) if c in "aeiou" and i > 0]
            if vs:
                i = random.choice(vs); w = w[:i] + w[i+1:]
        elif len(w) > 3 and r < 0.20:             # do akshar aage-peeche
            i = random.randrange(len(w) - 1); w = w[:i] + w[i+1] + w[i] + w[i+2:]
        elif r < 0.28:                            # alag spelling
            w = (w.replace("v", "w") if "v" in w else
                 w.replace("sh", "s") if "sh" in w else w.replace("aa", "a"))
        elif r < 0.33:
            w = w + w[-1]                         # akshar double
        out.append(w)
    t = " ".join(out)
    return re.sub(r"[?.,!]+", "", t) if random.random() < 0.6 else t

if __name__ == "__main__":
    t = pd.read_csv("data/translated/translated_queries.csv")
    scripts = {"hi": sanscript.DEVANAGARI, "mr": sanscript.DEVANAGARI, "gu": sanscript.GUJARATI}
    for lang, sc in scripts.items():
        t[f"{lang}_roman"] = t[lang].map(lambda x: to_roman(x, sc))
        t[f"{lang}_roman_noisy"] = t[f"{lang}_roman"].map(add_noise)
    t.to_csv("data/translated/translated_queries_roman.csv", index=False, encoding="utf-8-sig")
    print("Saved", len(t), "rows")
    print(t[["hi", "hi_roman", "hi_roman_noisy"]].head(6).to_string())
    print(t[["gu", "gu_roman", "gu_roman_noisy"]].head(4).to_string())
    print(t[["mr", "mr_roman", "mr_roman_noisy"]].head(4).to_string())
