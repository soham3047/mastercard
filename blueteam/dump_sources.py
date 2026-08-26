"""
Dumps ONLY b7_score_batch.py, in two halves if needed, to avoid truncation.
Run from D:\mastercard_hack\vineyard\blueteam
    python dump_b7_only.py
Paste the ENTIRE output back -- both PART 1 and PART 2.
"""
path = "src/b7_score_batch.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

lines = content.splitlines(keepends=True)
mid = len(lines) // 2

print("#" * 80)
print(f"# {path} -- PART 1 of 2 (lines 1-{mid})")
print("#" * 80)
print("".join(lines[:mid]))

print()
print("#" * 80)
print(f"# {path} -- PART 2 of 2 (lines {mid+1}-{len(lines)})")
print("#" * 80)
print("".join(lines[mid:]))

print()
print(f"[info] total lines: {len(lines)}, total chars: {len(content)}")