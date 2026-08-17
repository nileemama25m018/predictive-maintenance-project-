from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents(root: str | Path) -> list[dict]:
    root = Path(root)
    docs = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, chunk in enumerate(chunk_text(text)):
                docs.append({"id": f"{path}:{i}", "source": str(path), "chunk": i, "text": chunk})
    return docs


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(clean) <= max_chars:
        return [clean]
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


class RagIndex:
    def __init__(self, index_dir: str | Path = "rag_index"):
        self.index_dir = Path(index_dir)
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.docs: list[dict] = []

    def build(self, docs_root: str | Path) -> None:
        self.docs = load_documents(docs_root)
        if not self.docs:
            raise ValueError(f"No .txt/.md documents found under {docs_root}")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in self.docs])

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, self.index_dir / "vectorizer.joblib")
        joblib.dump(self.matrix, self.index_dir / "matrix.joblib")
        (self.index_dir / "docs.json").write_text(json.dumps(self.docs, indent=2), encoding="utf-8")

    def load(self) -> "RagIndex":
        self.vectorizer = joblib.load(self.index_dir / "vectorizer.joblib")
        self.matrix = joblib.load(self.index_dir / "matrix.joblib")
        self.docs = json.loads((self.index_dir / "docs.json").read_text(encoding="utf-8"))
        return self

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.vectorizer is None or self.matrix is None:
            self.load()
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(scores)[::-1][:k]
        return [
            {
                "score": float(scores[i]),
                "source": self.docs[i]["source"],
                "chunk": self.docs[i]["chunk"],
                "text": self.docs[i]["text"],
            }
            for i in order
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="knowledge_base")
    parser.add_argument("--index", default="rag_index")
    args = parser.parse_args()
    index = RagIndex(args.index)
    index.build(args.docs)
    index.save()
    print(f"Indexed {len(index.docs)} chunks into {args.index}")


if __name__ == "__main__":
    main()

