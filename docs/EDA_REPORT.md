# Exploratory Data Analysis (EDA) Report

Generated automatically by `scholarmind.data.eda.DatasetEDAAnalyzer`.

## Dataset: `arxiv_metadata`

- **Total Analyzed Rows:** 404
- **Exact Duplicate Rows:** 0

### Text Length Statistics (Word Count & Estimated Subword Tokens)

| Field | Mean Words | Median (P50) | P95 Words | Max Words | >512 Tok (%) | >4096 Tok (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `title` | 10.3 | 10.0 | 16.0 | 23.0 | 0.0% | 0.0% |
| `abstract` | 199.78 | 199.5 | 274.0 | 333.0 | 0.0% | 0.0% |

### Missing & Empty Value Summary

| Column | Dtype | Nulls | Empty/Whitespace | Missing % |
| :--- | :--- | ---: | ---: | ---: |
| `paper_id` | `str` | 0 | 0 | 0.0% |
| `title` | `str` | 0 | 0 | 0.0% |
| `abstract` | `str` | 0 | 0 | 0.0% |
| `authors` | `str` | 0 | 0 | 0.0% |
| `categories` | `str` | 0 | 0 | 0.0% |
| `target_categories` | `object` | 0 | 0 | 0.0% |
| `primary_category` | `str` | 0 | 0 | 0.0% |
| `num_categories` | `int64` | 0 | 0 | 0.0% |
| `year` | `int64` | 0 | 0 | 0.0% |
| `doi` | `str` | 0 | 404 | 100.0% |
| `journal_ref` | `str` | 0 | 404 | 100.0% |

### Key Distributions & Domain Insights

- **`primary_category_counts`:** `{'cs.LG': 75, 'cs.CV': 67, 'cs.AI': 54, 'cs.IR': 46, 'cs.RO': 46, 'cs.CL': 43, 'stat.ML': 37, 'cs.NE': 36}`
- **`avg_categories_per_paper`:** `2.07`
- **`multi_label_pct`:** `68.32`
- **`year_range`:** `[2026, 2026]`
- **`top_years`:** `{2026: 404}`

### Sample Record Preview
```json
{
  "paper_id": "2611.23229",
  "title": "Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching",
  "abstract": "Collecting large-scale, diverse, and high-quality post-training datasets for autoregressive language models is challenging. Consequently, Supervised Fine-Tuning (SFT) is typically applied to datasets significantly smaller than those used for pre-training, whic...",
  "authors": "Giannis Daras, Marios Papachristou, Adam Klivans, Antonio Torralba, Constantinos Daskalakis",
  "categories": "cs.LG cs.AI cs.CL",
  "target_categories": [
    "cs.LG",
    "cs.AI",
    "cs.CL"
  ],
  "primary_category": "cs.LG",
  "num_categories": 3,
  "year": 2026,
  "doi": "",
  "journal_ref": ""
}
```

## Dataset: `squad_v2`

- **Total Analyzed Rows:** 700
- **Exact Duplicate Rows:** 0
- **Split Sizes:** `{'train': 500, 'val': 200}`

### Text Length Statistics (Word Count & Estimated Subword Tokens)

| Field | Mean Words | Median (P50) | P95 Words | Max Words | >512 Tok (%) | >4096 Tok (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `context` | 127.94 | 117.0 | 253.2 | 402.0 | 0.29% | 0.0% |
| `question` | 10.28 | 10.0 | 17.0 | 31.0 | 0.0% | 0.0% |
| `answer_text_answerable_only` | 2.96 | 2.0 | 9.0 | 23.0 | 0.0% | 0.0% |

### Missing & Empty Value Summary

| Column | Dtype | Nulls | Empty/Whitespace | Missing % |
| :--- | :--- | ---: | ---: | ---: |
| `id` | `str` | 0 | 0 | 0.0% |
| `title` | `str` | 0 | 0 | 0.0% |
| `context` | `str` | 0 | 0 | 0.0% |
| `question` | `str` | 0 | 0 | 0.0% |
| `answer_text` | `str` | 0 | 229 | 32.71% |
| `answer_start` | `int64` | 0 | 0 | 0.0% |
| `all_answers` | `object` | 0 | 0 | 0.0% |
| `is_impossible` | `bool` | 0 | 0 | 0.0% |

### Key Distributions & Domain Insights

- **`unanswerable_pct_overall`:** `32.71`
- **`unanswerable_pct_by_split`:** `{'train': 27.4, 'val': 46.0}`
- **`unique_titles`:** `28`
- **`avg_questions_per_title`:** `25.0`

### Sample Record Preview
```json
{
  "id": "5ad2c8b1d7d075001a42a20b",
  "title": "Northwestern_University",
  "context": "In April 2016, Northwestern announced that it signed on to the Chicago Star Partnership, a City Colleges initiative. Through this partnership, Northwestern is one of 15 Illinois public and private universities that will \"provide scholarships to students who gr...",
  "question": " Who encouraged local universities to decrease opportunities for students in the public school districts?",
  "answer_text": "",
  "answer_start": -1,
  "all_answers": [],
  "is_impossible": true
}
```

## Dataset: `arxiv_summarization`

- **Total Analyzed Rows:** 160
- **Exact Duplicate Rows:** 0
- **Split Sizes:** `{'train': 100, 'val': 30, 'test': 30}`

### Text Length Statistics (Word Count & Estimated Subword Tokens)

| Field | Mean Words | Median (P50) | P95 Words | Max Words | >512 Tok (%) | >4096 Tok (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `article` | 6244.43 | 4906.5 | 14504.9 | 33095.0 | 100.0% | 78.75% |
| `abstract` | 215.72 | 166.0 | 328.0 | 2961.0 | 4.38% | 0.0% |

### Missing & Empty Value Summary

| Column | Dtype | Nulls | Empty/Whitespace | Missing % |
| :--- | :--- | ---: | ---: | ---: |
| `paper_id` | `str` | 0 | 0 | 0.0% |
| `article` | `str` | 0 | 0 | 0.0% |
| `abstract` | `str` | 0 | 0 | 0.0% |

### Key Distributions & Domain Insights

- **`compression_ratio_mean`:** `39.56`
- **`compression_ratio_median`:** `33.69`
- **`pct_articles_with_xmath`:** `99.38`
- **`pct_articles_with_xcite`:** `83.75`
- **`pct_articles_with_xref`:** `0.0`

### Sample Record Preview
```json
{
  "paper_id": "arxiv_summ_train_0",
  "article": "additive models @xcite provide an important family of models for semiparametric regression or classification . some reasons for the success of additive models are their increased flexibility when compared to linear or generalized linear models and their increa...",
  "abstract": "additive models play an important role in semiparametric statistics . \n this paper gives learning rates for regularized kernel based methods for additive models . \n these learning rates compare favourably in particular in high dimensions to recent results on o..."
}
```

## Dataset: `sciq`

- **Total Analyzed Rows:** 700
- **Exact Duplicate Rows:** 0
- **Split Sizes:** `{'train': 500, 'val': 100, 'test': 100}`

### Text Length Statistics (Word Count & Estimated Subword Tokens)

| Field | Mean Words | Median (P50) | P95 Words | Max Words | >512 Tok (%) | >4096 Tok (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `question` | 13.3 | 12.0 | 22.0 | 47.0 | 0.0% | 0.0% |
| `correct_answer` | 1.47 | 1.0 | 3.0 | 7.0 | 0.0% | 0.0% |
| `support_when_present` | 82.33 | 58.0 | 263.05 | 497.0 | 0.65% | 0.0% |

### Missing & Empty Value Summary

| Column | Dtype | Nulls | Empty/Whitespace | Missing % |
| :--- | :--- | ---: | ---: | ---: |
| `id` | `str` | 0 | 0 | 0.0% |
| `question` | `str` | 0 | 0 | 0.0% |
| `correct_answer` | `str` | 0 | 0 | 0.0% |
| `distractor1` | `str` | 0 | 0 | 0.0% |
| `distractor2` | `str` | 0 | 0 | 0.0% |
| `distractor3` | `str` | 0 | 0 | 0.0% |
| `support` | `str` | 0 | 80 | 11.43% |
| `has_support` | `bool` | 0 | 0 | 0.0% |

### Key Distributions & Domain Insights

- **`has_support_pct`:** `88.57`
- **`missing_support_count`:** `80`
- **`unique_support_passages`:** `618`
- **`avg_correct_answer_words`:** `1.47`
- **`avg_distractor1_words`:** `1.44`

### Sample Record Preview
```json
{
  "id": "sciq_train_00000",
  "question": "What type of organism is commonly used in preparation of foods such as cheese and yogurt?",
  "correct_answer": "mesophilic organisms",
  "distractor1": "protozoa",
  "distractor2": "gymnosperms",
  "distractor3": "viruses",
  "support": "Mesophiles grow best in moderate temperature, typically between 25°C and 40°C (77°F and 104°F). Mesophiles are often found living in or on the bodies of humans or other animals. The optimal growth temperature of many pathogenic mesophiles is 37°C (98°F), the n...",
  "has_support": true
}
```
