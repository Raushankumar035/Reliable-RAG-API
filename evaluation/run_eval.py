"""Run the evaluation question set against a *running* API and print metrics.

Usage:
    python -m evaluation.run_eval --ingest               # upload sample docs first, then evaluate
    python -m evaluation.run_eval --base-url http://localhost:8000 --delay 2

Metrics
  decision_accuracy : answered when it should / abstained when it should
  keyword_pass      : answerable questions whose answer contains all expected keywords
                      ("a|b" = either alternative accepted)
  source_recall     : answerable questions where every expected document appears in the cited sources
  hallucination     : unanswerable questions that the system nevertheless answered (lower is better)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT.parent / "data" / "sample_documents"


def keywords_ok(answer: str, keywords: list[str]) -> bool:
    text = answer.lower()
    return all(any(alt.strip().lower() in text for alt in kw.split("|")) for kw in keywords)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--questions", default=str(ROOT / "questions.json"))
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between questions (Groq free-tier rate limits)")
    parser.add_argument("--ingest", action="store_true", help="upload data/sample_documents first")
    parser.add_argument("--output", default=str(ROOT / "results.json"))
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    client = httpx.Client(base_url=args.base_url, timeout=120)

    if args.ingest:
        files = [("files", (p.name, p.read_bytes())) for p in sorted(SAMPLES.glob("*")) if p.is_file()]
        r = client.post("/documents", files=files)
        print("Ingest:", [(x["file_name"], x["status"]) for x in r.json().get("results", [])])

    rows, errors = [], 0
    for q in questions:
        try:
            r = client.post("/ask", json={"question": q["question"]})
            r.raise_for_status()
            body = r.json()
        except Exception as exc:
            errors += 1
            print(f"{q['id']} ERROR {exc}")
            rows.append({**q, "error": str(exc)})
            continue
        cited = {s["document"] for s in body["sources"]}
        row = {
            "id": q["id"], "category": q["category"], "question": q["question"],
            "expected_answerable": q["answerable"], "got_answerable": body["answerable"],
            "reason": body["reason"], "answer": body["answer"], "cited_documents": sorted(cited),
            "decision_correct": body["answerable"] == q["answerable"],
        }
        if q["answerable"]:
            row["keywords_ok"] = body["answerable"] and keywords_ok(body["answer"], q["expected_keywords"])
            row["sources_ok"] = set(q["expected_sources"]) <= cited
        rows.append(row)
        flag = "OK " if row["decision_correct"] and row.get("keywords_ok", True) else "BAD"
        print(f"{flag} {q['id']} [{q['category']:<14}] answerable={body['answerable']!s:<5} reason={body['reason']}")
        time.sleep(args.delay)

    scored = [r for r in rows if "error" not in r]
    ans = [r for r in scored if r["expected_answerable"]]
    unans = [r for r in scored if not r["expected_answerable"]]
    pct = lambda n, d: f"{n}/{d} ({(100 * n / d if d else 0):.0f}%)"
    summary = {
        "decision_accuracy": pct(sum(r["decision_correct"] for r in scored), len(scored)),
        "keyword_pass": pct(sum(r["keywords_ok"] for r in ans), len(ans)),
        "source_recall": pct(sum(r["sources_ok"] for r in ans), len(ans)),
        "hallucination": pct(sum(r["got_answerable"] for r in unans), len(unans)),
        "request_errors": errors,
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>18}: {v}")
    Path(args.output).write_text(json.dumps({"summary": summary, "results": rows}, indent=2))
    print(f"\nDetails written to {args.output}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
