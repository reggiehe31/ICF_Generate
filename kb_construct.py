import argparse, json, os, time
from pathlib import Path

import chromadb
import fitz                      
import requests
import yaml
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}

SEPARATORS = ["\n\n\n", "\n\n", "\n", "。", "！", "？", "；",
              ". ", "! ", "? ", "; ", "，", ", ", " ", ""]


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="ignore")
        if ext == ".pdf":
            with fitz.open(path) as doc:
                return "\n\n".join(p.get_text("text").strip() for p in doc
                                   if p.get_text("text").strip())
        if ext == ".docx":
            return "\n".join(p.text for p in Document(path).paragraphs)
    except Exception as e:
        print(f"  ! read failed: {e}")
    return ""


def embed_batch(texts, url, retries=3, timeout=180):
    for attempt in range(retries):
        try:
            r = requests.post(url, json={"input": texts, "model": "bge-m3"}, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            data = body["data"] if isinstance(body, dict) and "data" in body else body
            return [d["embedding"] for d in data]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--kb-dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    kcfg, scfg = cfg["knowledge_base"], cfg["servers"]
    work = Path(cfg["paths"]["work_dir"])
    kb_dir, chroma_dir = Path(args.kb_dir), work / cfg["paths"]["chroma_dir"]
    record_file = work / "processed_files.json"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=kcfg["chunk_size"], chunk_overlap=kcfg["chunk_overlap"],
        separators=SEPARATORS, length_function=len, keep_separator=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_or_create_collection(
        kcfg["collection"],
        metadata={"hnsw:space": kcfg["distance_metric"]},   # explicit; do not omit
    )
    record = json.loads(record_file.read_text(encoding="utf-8")) if record_file.exists() else {}
    print(f"collection '{kcfg['collection']}' ({kcfg['distance_metric']}): {col.count()} chunks")

    added_total = 0
    for path in sorted(p for p in kb_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in SUPPORTED
                       and not p.name.startswith("~$")):
        key, mtime = str(path), str(path.stat().st_mtime)
        if record.get(key) == mtime:
            continue
        print(f"{path.name} ...", end=" ", flush=True)

        text = read_file(path)
        chunks = [c for c in splitter.split_text(text)
                  if len(c.strip()) >= kcfg["min_chunk_chars"]] if text.strip() else []
        if not chunks:
            print("empty, skipped")
            record[key] = mtime
            continue

        old = col.get(where={"source": path.name})
        if old["ids"]:
            col.delete(ids=old["ids"])

        if kcfg["context_injection"]:
            stem = path.stem
            for suffix in ("指南", "规范", "知情同意书", "专家共识"):
                stem = stem.replace(suffix, "")
            chunks = [f"【所属文献：{path.name} | 相关术式：{stem}】\n{c}" for c in chunks]

        added = 0
        for i in range(0, len(chunks), kcfg["embed_batch_size"]):
            batch = chunks[i:i + kcfg["embed_batch_size"]]
            try:
                vecs = embed_batch(batch, scfg["embed_url"], scfg["retries"], scfg["timeout"])
                col.add(ids=[f"{path.name}_{i + j}" for j in range(len(batch))],
                        embeddings=vecs, documents=batch,
                        metadatas=[{"source": path.name, "chunk_idx": i + j}
                                   for j in range(len(batch))])
                added += len(batch)
            except Exception as e:
                print(f"\n  ! batch {i} failed: {e}")

        record[key] = mtime
        added_total += added
        print(f"{added} chunks")

    record_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nadded {added_total}; collection now holds {col.count()} chunks")


if __name__ == "__main__":
    main()
