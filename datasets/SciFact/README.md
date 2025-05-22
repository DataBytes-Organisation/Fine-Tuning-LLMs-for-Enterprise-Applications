
# SciFact Dataset Preprocessing Documentation
**DataBytes | Fine-Tuning LLMs for Enterprise Applications | Medical Misinformation Detection in LLM Responses**

*Manusha Fernando | S223259359 | s223259359@deakin.edu.au*

Source: https://huggingface.co/datasets/allenai/scifact

---

## 1. Objective

Prepare a structured dataset to fine-tune a language model to classify scientific claims as:
- **True** (supported by evidence)
- **False** (contradicted by evidence)
- **Misleading** (no evidence provided)

---

## 2. Source Files

| File               | Description                                |
|--------------------|--------------------------------------------|
| `claims_train.jsonl` | Training claims + evidence mappings      |
| `claims_dev.jsonl`   | Validation claims + evidence mappings    |
| `claims_test.jsonl`  | Test claims, **no gold labels**          |
| `corpus.jsonl`       | Abstracts used as evidence source        |

---

## 3. Dataset Structure

Each claim entry includes:
- `claim`: A scientific claim (string)
- `evidence`: A dict of `doc_id: [annotations]` with sentence indices and label (`SUPPORT`, `CONTRADICT`)
- `cited_doc_ids`: List of relevant document IDs

Each corpus entry includes:
- `doc_id`: Document identifier
- `abstract`: List of abstract sentences

---

## 4. Preprocessing Steps

### Corpus Lookup
Created a mapping from `doc_id` to joined abstract text for fast evidence retrieval.

### Label Mapping

| Original Label   | Mapped Label |
|------------------|--------------|
| `SUPPORT`        | `True`       |
| `CONTRADICT`     | `False`      |
| (No evidence)    | `Misleading` |

### Evidence Extraction
- For entries with evidence:
  - Retrieved `abstract[sent_index]` for each annotation
  - Concatenated evidence sentences into `evidence_text`
- For entries **without evidence**:
  - Assigned label `Misleading`
  - Set `evidence_text` to empty string

---

## 5. Final Dataset Format

Each example contains:

```json
{
  "claim": "string",
  "evidence_text": "string",
  "label": "True" | "False" | "Misleading"
}
```

Saved as:
- `train_3class.jsonl`
- `dev_3class.jsonl`

---
## 6. Output Summary

| Dataset     | Total Samples | True | False | Misleading |
|-------------|----------------|------|-------|------------|
| Train       | 809            | TBD  | TBD   | 304        |
| Dev         | 300            | TBD  | TBD   | 112        |
| Test        | 300            | N/A  | N/A   | N/A        |

> Use `train_3class.jsonl` for training the model.
> Use `dev_3class.jsonl` for testing the model.
> Use `claims_test.jsonl` for inference only — no ground-truth labels provided.

---

## 7. Usage Notes

- Suitable for fine-tuning models.
- Prompt-based inference ready (e.g., "Is this claim true, false, or misleading?").
- Compatible with Hugging Face tokenizer/pipeline workflows.
