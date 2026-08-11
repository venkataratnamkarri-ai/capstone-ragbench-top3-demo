import json, glob, os

root = os.getcwd()
files = sorted(glob.glob('**/*.jsonl', recursive=True))
results = []
for f in files:
    vals = []
    with open(f, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            metrics = [obj.get('pred_relevance'), obj.get('pred_utilization'), obj.get('pred_completeness'), obj.get('pred_adherence')]
            if all(v is not None for v in metrics):
                vals.append(sum(metrics) / len(metrics))
    if vals:
        avg = sum(vals) / len(vals)
        results.append((avg, len(vals), f))
results.sort(reverse=True)
print('TOP3')
for avg, n, f in results[:3]:
    print(f'{avg:.4f}\t{n}\t{f}')
