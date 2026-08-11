# ICF_Generate

Generation pipeline for personalised Chinese surgical informed-consent documents,
with and without retrieval augmentation.

This is the code behind the V2 (personalised) and V3 (RAG-augmented) conditions
described in the Methods. The two conditions run through the same script and the
same prompt file; they differ only in whether a retrieved knowledge block is
present. `render_prompts.py` writes both prompts to disk and diffs them, so the
difference between conditions is auditable before you spend any API budget.

Institutional patient records, consent templates and the knowledge base are not
included and cannot be redistributed. You supply your own.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your API key
```

V3 needs two local llama.cpp servers:

```bash
llama-server -m models/bge-m3-Q8_0.gguf        --embedding --port 18080 -c 8192 -b 512
llama-server -m models/bge-reranker-v2-m3.gguf --reranking --port 18081
```

## Configure

Edit `config.yaml`:

- `models:` — add one block per backbone. The key is a label you choose and pass
  to `--model`; each block needs `api_model_string`, `base_url` (any
  OpenAI-compatible endpoint) and `api_key_env` (the name of the variable in
  `.env` holding that key).
- `paths.templates:` — map each consent-type label to a template PDF in
  `work_dir`. The labels must match the values in your cohort file's consent-type
  column, or the template will not be found.
- `paths.cohort_file:` — an Excel file, one row per patient. Column names are
  declared under `fields:`; the whitelist under `patient_fields:` controls what
  reaches the model.

Everything else has working defaults: chunk size 512/64, dense top-20 per facet
query, top-2 retained after reranking, top-6 in the final knowledge block,
temperature 0.3, top-p 0.9.

## Run

Build the knowledge base once:

```bash
python kb_construct.py --config config.yaml --kb-dir ./data/knowledge_base
```

Ingests `.txt`, `.md`, `.pdf`, `.docx`. Spreadsheets are skipped on purpose, so
a cohort file sitting in the same tree cannot leak into the index.

Check what the prompts look like:

```bash
python render_prompts.py
```

Writes `rendered_prompts/prompt_v2.md`, `prompt_v3.md` and `diff_v2_v3.txt`.
Read the diff. If it shows anything beyond the RAG fragments, the controlled
contrast is broken.

Generate:

```bash
python icf_generate.py --version v2 --model my_model
python icf_generate.py --version v3 --model my_model --limit 5   # smoke test first
```

Output goes to `data/generated_<version>_<model>/`, one UTF-8 text file per
patient. V3 also writes `data/retrieval_logs_v3_<model>/` with the facet queries,
candidate counts, rerank scores and retained passages for each document. Existing
output files are skipped, so an interrupted batch resumes where it stopped.

## Retrieval

Queries come from each patient's structured profile, not from a fixed template.
Up to six facet queries are built: core-procedure complications, procedural steps
and prevention, diagnosis and prognosis, surgical indication, an age-conditioned
query (menopause and ovarian preservation at 50 and above, fertility below 40),
and a peri-operative query. The last four are emitted only when the corresponding
patient field is present, so the number of active queries varies by patient.

Each query is retrieved and reranked on its own and keeps its top 2 passages
before pooling. One high-scoring topic therefore cannot crowd the others out.

Retrieval runs as a separate pass for each generation, but the queries are
derived deterministically from the patient profile: the same patient yields the
same evidence block regardless of which backbone consumes it.

## Judge prompts

`prompts/judge/` holds the verbatim Chinese prompts issued to the LLM judge for
the KSGO 16-item content-completeness instrument and the 20-item risk-disclosure
specificity instrument, with all scoring anchors expanded. Judge calls ran at
temperature 0, one call per document. English element definitions and anchors are
in Supplementary Tables S1a and S1b.

## Layout

```
config.yaml              paths, fields, retrieval and generation parameters
kb_construct.py          BGE-M3 + ChromaDB index construction
icf_generate.py          generation driver, retrieval included
prompt_builder.py        assembles a condition's prompt from base + fragments
render_prompts.py        dumps both prompts and diffs them
prompts/base_prompt.md   shared skeleton, with {{RAG_*}} slots
prompts/fragments/       what fills those slots in each condition
prompts/judge/           scoring prompts for the two evaluation instruments
```

## Licence

MIT.
