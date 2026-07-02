"""
Quick, cheap few-shot numeracy/reasoning probe of BASE models (no SFT/DPO/CoT).

The whole §10 finding is that reasoning is gated by the base model's numeracy.
So to get a *rough* read on whether the FineMath midtraining checkpoint improved
reasoning, we few-shot the raw bases on arithmetic + multi-step problems and
compare accuracy. No instruction tuning needed -- base LMs answer few-shot.

Loads each base via ArgonneModel.from_pretrained (the self-healing path that
rebuilds RoPE buffers + ties lm_head), greedy-decodes, extracts the final number.
"""

import re
import sys
import torch

sys.path.insert(0, "/home/youzhi/ArgonneAI")
from model import ArgonneModel
from transformers import AutoTokenizer

BASES = [
    ("argonne-3.0-base (pre-midtrain)", "/project/rcc/youzhi/models/pretrain/argonne-3.0-base"),
    ("midtrain longmino (Phase1)",      "/project/rcc/youzhi/models/midtrain/final_model_complete_longmino"),
    ("midtrain_finemath (Phase2, NEW)", "/project/rcc/youzhi/models/midtrain_finemath_pinned/final_model_complete"),
]

# 4-shot GSM8K-style exemplars (kept short; teach the "#### <answer>" convention).
FEWSHOT = """Question: What is 6 times 7?
Answer: 6 times 7 is 42. #### 42

Question: Tom has 12 apples and gives away 5. How many are left?
Answer: 12 minus 5 is 7. #### 7

Question: What is half of 80?
Answer: Half of 80 is 80 divided by 2, which is 40. #### 40

Question: If 3x = 21, what is x?
Answer: x is 21 divided by 3, which is 7. #### 7

"""

# (question, gold) -- mix of single-fact arithmetic and multi-step reasoning.
PROBES = [
    ("What is 8 plus 3?", 11),
    ("What is 17 minus 5?", 12),
    ("What is 100 divided by 4?", 25),
    ("What is 9 times 6?", 54),
    ("What is 144 divided by 12?", 12),
    ("What is 7 plus 8?", 15),
    ("If 2x + 5 = 17, what is x?", 6),
    ("What is the sum of all integers from 1 to 10?", 55),
    ("How many positive divisors does 12 have?", 6),
    ("A rectangle has length 8 and width 3. What is its perimeter?", 22),
    ("What is 13 times 4?", 52),
    ("Sarah has 3 boxes with 6 pencils each. How many pencils total?", 18),
    ("A train travels 60 miles per hour for 3 hours. How far does it go?", 180),
    ("What is 25 percent of 200?", 50),
    ("If a shirt costs $20 and is 10% off, what is the sale price?", 18),
    ("What is 5 factorial (5!)?", 120),
    ("What is the next prime number after 7?", 11),
    ("John is twice as old as Mary. Mary is 9. How old is John?", 18),
    ("What is 1000 minus 256?", 744),
    ("A dozen eggs costs $3. How much do 4 dozen cost?", 12),
]

def extract_answer(text):
    # Prefer the number right after the first '####'; else the last integer.
    m = re.search(r"####\s*(-?\d[\d,]*)", text)
    if m:
        return int(m.group(1).replace(",", ""))
    nums = re.findall(r"-?\d[\d,]*", text)
    return int(nums[-1].replace(",", "")) if nums else None

@torch.no_grad()
def run_base(name, path, tok):
    print(f"\n{'='*70}\n{name}\n  {path}\n{'='*70}", flush=True)
    model = ArgonneModel.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()
    correct = 0
    rows = []
    for q, gold in PROBES:
        prompt = FEWSHOT + f"Question: {q}\nAnswer:"
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        out = model.generate(ids, max_length=ids.shape[1] + 80, do_sample=False)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        # Only look at the first line of the answer (stop at next "Question:").
        gen_line = gen.split("Question:")[0].strip()
        pred = extract_answer(gen_line)
        ok = (pred == gold)
        correct += ok
        rows.append((ok, q, gold, pred, gen_line.replace("\n", " ")[:80]))
    for ok, q, gold, pred, snip in rows:
        print(f"  [{'Y' if ok else 'n'}] {q[:48]:48s} gold={gold:<5} pred={str(pred):<6} | {snip}", flush=True)
    print(f"  --> {name}: {correct}/{len(PROBES)} correct", flush=True)
    del model
    torch.cuda.empty_cache()
    return correct

def main():
    tok = AutoTokenizer.from_pretrained(BASES[0][1], trust_remote_code=True)
    results = []
    for name, path in BASES:
        try:
            results.append((name, run_base(name, path, tok)))
        except Exception as e:
            print(f"  FAILED {name}: {e}", flush=True)
            results.append((name, -1))
    print(f"\n{'#'*70}\nSUMMARY (few-shot, {len(PROBES)} problems)\n{'#'*70}")
    for name, score in results:
        print(f"  {score}/{len(PROBES)}   {name}")

if __name__ == "__main__":
    main()
