"""Score a saved round results file and print NDCG@4 per question type.

Usage: python score_round.py <results.json> <qa_dataset.jsonl> <round_label> [desc]

results.json format: {qid: result} (app_dir/eval_results.json)
                 OR: {mode|qid: result} (combined new_vision_modes_eval_results*.json)
"""
import json, sys
from pathlib import Path

results_file = Path(sys.argv[1])
qa_file = Path(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else results_file.stem
desc = sys.argv[4] if len(sys.argv) > 4 else ""

data_raw = json.loads(results_file.read_text())
qa_items = [json.loads(l) for l in qa_file.read_text().splitlines() if l.strip()]
qa_map = {q['id']: q for q in qa_items}


def ndcg(retrieved, relevant, k=4):
    hits = retrieved[:k]
    def dcg(lst): return sum(1 / (i + 1) for i, p in enumerate(lst) if p in relevant)
    ideal = min(len(relevant), k)
    return dcg(hits) / dcg(list(relevant)[:ideal]) if ideal else 0.0


def rel(q):
    docs, pages = q.get('relevant_docs', []), q.get('relevant_pages', [])
    if not docs or not pages:
        return set()
    if len(docs) == 1:
        return {(docs[0], int(p)) for p in pages}
    return {(d, int(p)) for d, p in zip(docs, pages)}


def retrieved(r):
    pages = []
    for h in r.get('image_hits', []) + r.get('text_hits', []):
        d, p = h.get('doc', ''), h.get('page')
        if d and p is not None:
            pages.append((d, int(p)))
    seen = []
    [seen.append(x) for x in pages if x not in seen]
    return seen[:4]


# Support both {qid: result} and {mode|qid: result} formats
def iter_results():
    for k, v in data_raw.items():
        if '|' in k:
            mode, qid = k.split('|', 1)
            if mode != 'vision_hybrid_bm25':
                continue
        else:
            qid = k
        yield qid, v


counts = {'cross_doc': 8, 'cross_page': 3, 'exact_lookup': 19,
          'semantic_paraphrase': 3, 'table_cell': 35, 'visual_only': 47}
scores = {t: [] for t in counts}

for qid, v in iter_results():
    q = qa_map.get(qid)
    if not q:
        continue
    r = rel(q)
    if not r:
        continue
    t = q.get('question_type', '')
    if t in scores:
        scores[t].append(ndcg(retrieved(v), r))

avgs = {t: (sum(v) / len(v) if v else 0.0) for t, v in scores.items()}
total = sum(avgs[t] * counts[t] for t in counts) / sum(counts.values())

desc_str = f"  [{desc}]" if desc else ""
print(f"R{label} NDCG@4 = {total:.3f}{desc_str}")
for t in ['visual_only', 'exact_lookup', 'table_cell', 'cross_doc', 'semantic_paraphrase', 'cross_page']:
    n = len(scores[t])
    print(f"  {t:<24} {avgs[t]:.3f}  (n={n})")
