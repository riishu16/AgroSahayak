# Data Sources

All data used in this project is public. For every source, this file records the name, link, download date, licence/terms, and what was used.

## 1. Datasets used

| # | Source | Link | Download date | Licence / Terms | What was used |
|---|---|---|---|---|---|
| 1 | Kisan Call Centre (KCC) transcripts, Bihar | https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers | 26-Nov-2026 | Government Open Data License - India (GODL), as stated on data.gov.in. Usage selected at download: Non-Commercial, Academia | Bihar, Jan 2026, 31,720 records |
| 2 | Kisan Call Centre (KCC) transcripts, Gujarat | https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers | 26-Nov-2026 | Government Open Data License - India (GODL), as stated on data.gov.in. Usage selected at download: Non-Commercial, Academia | Gujarat, Jan 2026, 6,677 records |
| 3 | Kisan Call Centre (KCC) transcripts, Maharashtra | https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers | 26-Nov-2026 | Government Open Data License - India (GODL), as stated on data.gov.in. Usage selected at download: Non-Commercial, Academia | Maharashtra, Jan 2026, 7,931 records |

The raw files are stored in `data/raw/` and are not pushed to GitHub (listed in `.gitignore`).

## 2. Models used (to build data)

| # | Model | Link | Date | Licence | Used for |
|---|---|---|---|---|---|
| 4 | IndicTrans2 En-Indic Distilled 200M (`ai4bharat/indictrans2-en-indic-dist-200M`) | https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M | 27-Nov-2026 | MIT (as shown on the model page). Gated model; access granted via Hugging Face | Machine translation of English KCC queries into Hindi, Gujarati and Marathi |

Note: the translated queries are synthetic, not real farmer text. The Gujarati and Marathi translations were not verified by a native speaker.

## 3. Derived files in this repository

| File | How it was made |
|---|---|
| `data/clean/kcc_clean.csv` | From the 3 raw state files: removed rows with missing labels, "INCOMPLETE CALL" and "Other State call" rows, "Other" crop and "NA" query type; normalized crop names; removed duplicates. Row counts: 46,328 -> 18,502 -> 6,883 -> 2,908 -> 2,773 |
| `data/final/train_pool.csv` | Built by `make_labels.py`: merged query types into 9 categories, relabeled "Disease (Protozoan)" rows using keyword rules, removed train rows too similar to the test pool (TF-IDF cosine similarity >= 0.8, 241 rows). 2,062 rows |
| `data/final/test_pool.csv` | Random pool of 150 rows (seed 42) |
| `data/final/test_final.csv` | 100 items from the test pool. Crop, category and entity labels were drafted with an AI tool (Claude) and then reviewed |
| `data/translated/translated_queries.csv` | Hindi, Gujarati and Marathi translations produced with IndicTrans2 |

## 4. Planned sources (not used yet)

- Package of Practices (state agricultural universities, KVK/ICAR advisories)
- Government scheme documents (PM-KISAN, PMFBY, Soil Health Card)
- AI4Bharat Dakshina (transliteration pairs)

A row will be added to the table in Section 1 as soon as any of these is used. Remove this section from the final submission if they are not used.

## 5. Attribution

Data source: Kisan Call Centre transcripts, Ministry of Agriculture & Farmers Welfare, Government of India, via data.gov.in (Open Government Data Platform India), licensed under GODL-India.