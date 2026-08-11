import argparse, json, os, sys, time
from pathlib import Path

import chromadb
import pandas as pd
import requests
import yaml
from openai import OpenAI
from pypdf import PdfReader

from prompt_builder import CONDITIONS, build_prompt

ROOT = Path(__file__).resolve().parent

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def embed(texts, url, timeout, retries):
    payload = {"input": texts, "model": "bge-m3"}
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            data = body["data"] if isinstance(body, dict) and "data" in body else body
            return [d["embedding"] for d in data]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def rerank(query, docs, url, top_n, timeout, retries):
    payload = {"model": "bge-reranker-v2-m3", "query": query,
               "documents": [d["text"] for d in docs], "top_n": top_n}
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            out = r.json().get("results", [])
            return [{"document": docs[o["index"]],
                     "score": o.get("relevance_score", o.get("score", 0.0))} for o in out]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def facet_queries(row, cfg) -> list:

    def get(col):
        v = row.get(col)
        return str(v).strip() if pd.notna(v) and str(v).strip() else ""

    surgery = get(cfg["fields"]["procedure"]) or get(cfg["fields"]["diagnosis"])
    diagnosis = get(cfg["fields"]["diagnosis"])
    indication = get(cfg["fields"]["indication"])
    contra = get(cfg["fields"]["contraindication"])
    age = pd.to_numeric(row.get(cfg["fields"]["age"]), errors="coerce")

    q = []
    if surgery:
        q.append(f"{surgery} 并发症 风险 发生率")          
        q.append(f"{surgery} 手术步骤 预防措施")            
    if diagnosis:
        q.append(f"{diagnosis} 治疗 预后")                  
    if surgery and indication:
        q.append(f"{surgery} 适应证 {indication}")         
    if surgery and pd.notna(age):                          
        if age >= cfg["retrieval"]["age_menopause_threshold"]:
            q.append(f"{surgery} 绝经期 卵巢保留")
        elif age < cfg["retrieval"]["age_fertility_threshold"]:
            q.append(f"{surgery} 生育功能 年轻女性")
    if surgery and contra:                                  
        q.append(f"{surgery} {contra} 围术期管理")
    return q


def retrieve(row, col, cfg) -> tuple:

    rcfg, scfg = cfg["retrieval"], cfg["servers"]
    queries = facet_queries(row, cfg)
    if not queries:
        return "（未检索到相关资料）", {"queries": [], "candidates_total": 0, "final": []}

    pool, n_cand = {}, 0
    for q in queries:
        try:
            vec = embed([q], scfg["embed_url"], scfg["timeout"], scfg["retries"])[0]
            res = col.query(query_embeddings=[vec], n_results=rcfg["top_k_retrieve"])
            cands = [{"id": cid, "text": res["documents"][0][i],
                      "metadata": (res["metadatas"][0][i] if res.get("metadatas") else {}),
                      "matched_query": q}
                     for i, cid in enumerate(res["ids"][0])]
            n_cand += len(cands)
            for item in rerank(q, cands, scfg["rerank_url"], rcfg["top_n_per_query"],
                               scfg["timeout"], scfg["retries"]):
                doc, score = item["document"], item["score"]
                if doc["id"] not in pool or score > pool[doc["id"]]["score"]:
                    pool[doc["id"]] = {"document": doc, "score": score}
        except Exception as e:
            print(f"    ! facet query failed ({q[:24]}...): {e}")

    if not pool:
        return "（未检索到相关资料）", {"queries": queries, "candidates_total": n_cand, "final": []}

    ranked = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[: rcfg["top_k_final"]]
    parts, log = [], []
    for i, item in enumerate(ranked, 1):
        d, s = item["document"], item["score"]
        src = (d.get("metadata") or {}).get("source", "unknown")
        parts.append(f"【参考{i} | 来源: {src} | 相关度: {s:.3f} | "
                     f"匹配维度: {d['matched_query']}】\n{d['text']}")
        log.append({"rank": i, "id": d["id"], "source": src, "rerank_score": s,
                    "matched_query": d["matched_query"], "text": d["text"]})
    return "\n\n".join(parts), {"queries": queries, "candidates_total": n_cand, "final": log}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(CONDITIONS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="cap N patients (smoke test)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.model not in cfg["models"]:
        sys.exit(f"unknown model '{args.model}'; configured: {list(cfg['models'])}")
    mcfg = cfg["models"][args.model]

    api_key = os.environ.get(mcfg["api_key_env"])
    if not api_key:
        sys.exit(f"environment variable {mcfg['api_key_env']} is not set")

    work = Path(cfg["paths"]["work_dir"])
    out_dir = work / f"generated_{args.version}_{args.model}"
    log_dir = work / f"retrieval_logs_{args.version}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.version == "v3":
        log_dir.mkdir(parents=True, exist_ok=True)

    templates = {}
    for name, fn in cfg["paths"]["templates"].items():
        p = work / fn
        if p.exists():
            templates[name] = "\n".join((pg.extract_text() or "") for pg in PdfReader(p).pages)
        else:
            print(f"! template missing: {p}")

    df = pd.read_excel(work / cfg["paths"]["cohort_file"])
    if args.limit:
        df = df.head(args.limit)

    col = None
    if args.version == "v3":
        client_db = chromadb.PersistentClient(path=str(work / cfg["paths"]["chroma_dir"]))
        col = client_db.get_collection(cfg["knowledge_base"]["collection"])
        print(f"knowledge base '{cfg['knowledge_base']['collection']}': {col.count()} chunks")

    prompt_template = build_prompt(args.version)
    llm = OpenAI(api_key=api_key, base_url=mcfg["base_url"])
    fields, gcfg = cfg["patient_fields"], cfg["generation"]

    print(f"condition={args.version}  backbone={mcfg['api_model_string']}  n={len(df)}")
    for idx, row in df.iterrows():
        pid = str(row.get(cfg["fields"]["patient_id"], f"row{idx}"))
        out_path = out_dir / f"{pid}+{args.version.upper()}.txt"
        if out_path.exists():
            continue

        ctype = str(row.get(cfg["fields"]["consent_type"], "")).strip()
        title = next((t for k, t in cfg["consent_titles"].items() if k in ctype),
                     cfg["consent_titles_default"])
        template_text = templates.get(ctype, "（通用框架）")

        patient_block = "\n".join(
            f"- {c}: {row[c]}" for c in fields
            if c in df.columns and pd.notna(row[c]) and str(row[c]).strip()
        )

        retrieved, rlog = "", None
        if args.version == "v3":
            retrieved, rlog = retrieve(row, col, cfg)

        user_msg = prompt_template.format(
            hospital_template=template_text,
            patient_data=patient_block,
            consent_title=title,
            retrieved_context=retrieved,
        )

        try:
            resp = llm.chat.completions.create(
                model=mcfg["api_model_string"],
                messages=[{"role": "system", "content": gcfg["system_prompt"]},
                          {"role": "user", "content": user_msg}],
                temperature=gcfg["temperature"],
                top_p=gcfg["top_p"],
            )
            out_path.write_text(resp.choices[0].message.content, encoding="utf-8")
            if rlog is not None:
                (log_dir / f"{pid}.json").write_text(
                    json.dumps(rlog, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{idx + 1}/{len(df)}] {out_path.name}")
            time.sleep(gcfg["sleep_seconds"])
        except Exception as e:
            print(f"  [{idx + 1}/{len(df)}] FAILED {pid}: {e}")

    print("done")


if __name__ == "__main__":
    main()
