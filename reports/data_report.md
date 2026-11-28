# AgroSahayak - Data Report

## 1. Data sources

| Source | Used for | Licence |
|---|---|---|
| Kisan Call Centre transcripts (data.gov.in) | Bihar, Gujarat, Maharashtra, Jan 2026 | GODL-India |
| IndicTrans2 En-Indic Distilled 200M (translation) | English -> Hindi, Gujarati, Marathi | MIT |

## 2. Cleaning pipeline

| Step | Rows |
|---|---|
| Raw KCC records (Bihar, Gujarat, Maharashtra; Jan 2026) | 46,328 |
| After removing rows with missing crop / query type / text / answer | 18,502 |
| After removing incomplete calls, 'Other' crop and 'NA' query type | 6,883 |
| After removing duplicates (normalized text + crop + query type) | 2,908 |
| After removing very short queries (< 3 words) | 2,773 |
| After mapping to 9 categories (rare / unmappable types dropped) | 2,453 |
| After removing near-duplicates of test pool (TF-IDF cosine >= 0.8) | 2,062 |

## 3. Train / test separation

| Check | Result |
|---|---|
| Near-duplicate removal (TF-IDF cosine >= 0.8) against test pool | 241 train rows removed |
| Exact text overlap train vs test, all languages and scripts (before) | 9 |
| Train examples removed because of exact overlap | 13 |
| Exact text overlap after removal | 0 |

## 4. Category distribution

| Category | Train rows | Test items (100) |
|---|---|---|
| cultivation | 188 | 10 |
| disease | 475 | 19 |
| insect_pest | 456 | 20 |
| irrigation | 77 | 5 |
| market | 30 | 3 |
| nutrient | 722 | 33 |
| post_harvest | 6 | 0 |
| variety_seed | 107 | 4 |
| weather | 1 | 6 |

Test items with crop = unknown (crop not named in the query): 9

## 5. Task 1 dataset (query -> JSON)

| Item | Value |
|---|---|
| Training examples | 8,297 |
|   of which synthetic (weather, post_harvest) | 62 |
|   with crop = unknown | 19.5% |
|   with a specific entity | 39.3% |
| Test examples (100 items x 7 variants) | 700 |

Examples by language and script:

| Language | Script | Train | Test |
|---|---|---|---|
| en | native | 2087 | 100 |
| gu | native | 1036 | 100 |
| gu | roman | 1007 | 100 |
| hi | native | 1036 | 100 |
| hi | roman | 1060 | 100 |
| mr | native | 1049 | 100 |
| mr | roman | 1022 | 100 |

## 6. Known limitations

- KCC QueryText is an operator-written English summary, not the farmer's own words.
- Hindi, Gujarati and Marathi queries are machine translations (IndicTrans2); Gujarati and Marathi were not verified by a native speaker.
- Roman-script queries are rule-based transliterations with random spelling noise, not real farmer text.
- Training entity labels are keyword-rule based; test entity labels were reviewed manually.
- Test labels were drafted with an AI tool (Claude) and reviewed.
- Weather and post_harvest had almost no training data after de-duplication; synthetic examples were added and are flagged.
- Only one month (Jan 2026) of three states was used.