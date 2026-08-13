#!/bin/bash
#SBATCH --job-name=exp-a4kd3
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --exclude=midway3-0423,midway3-0385,midway3-0602,midway3-[0298,0377-0378,0603-0606]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --time=07:00:00
#SBATCH --output=report/a4kd3_%j.out
#SBATCH --error=report/a4kd3_%j.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
# =============================================================================
# KEEP THE +7.6pp. The one arm on this base that ever moved `acc|ANSWERED` also broke termination, and
# the gain is being spent on the break. Two independent ways of protecting termination, plus the seed
# replicate that decides whether the gain is real at all.
#
# ⚠️TEACHER CHANGED from Qwen3-4B-Thinking-2507 to the released 3.5-think. The Qwen teacher carries 4.5x
# more per-token signal but its trace-length distribution is 20-30x the student's, and §41f measured that
# as greedy 1.75 with a 96.95% unclosed rate -- not repairable at decode time. 3.5-think is the teacher
# that actually produced the +7.6pp. `TEACHER=` still overrides if the Qwen arm is ever worth re-asking
# once `notail` has shown whether column-masking is sufficient protection.
#
# THE RESULT THIS IS BUILT ON (§41p). Per-token reverse KL from the length-matched released 3.5-think,
# on a4's own rollouts, took `acc|ANSWERED` from 61.2% to **68.8% (+7.6pp)** on asdiv+svamp -- the number
# thirteen previous arms left frozen, whose best was +2.5pp. Same sign on both pools (+5.1 / +10.2).
# Greedy moved only +1.70 because greedy = acc|ANSWERED x answered-rate, and the answered rate fell
# 92.7% -> 85.1% as unclosed doubled (7.15% -> 14.35%) and t_len went 200 -> 266.
#
#   68.8% acc|ANSWERED at combo's 92.7% answered rate would be greedy 63.8 -- +7.1 over baseline, more
#   than the entire thirteen-arm campaign produced. The capability gain is real and the shape regression
#   is eating it. This job tries to stop it being eaten, two independent ways, and replicates the seed.
#
# ARM `notail` -- SURGICAL. The `</think>` and eos COLUMNS are dropped from the divergence and both
#   distributions renormalised over content tokens only, so the teacher cannot move trace length at all.
#   Verified numerically: the excluded logits receive EXACTLY zero gradient from the KD term, kept
#   columns nonzero, and the loss differs from full-vocab so the mask is live.
#   ⚠️COLUMNS, not the positions whose target is a terminator -- the damage comes from the ~200 positions
#   per trace whose target is ordinary text and where the teacher still prefers to continue.
#
# ARM `anchor` -- STANDARD, as an independent check. Full-vocab reverse KL plus --ce-weight 0.5 on
#   gold-verified rows, pinning the model's own verified format with a likelihood term. If `notail` works
#   and `anchor` does not, the protection has to be structural; if both work, the effect is robust to how
#   it is protected.
#
# ARM `s47` -- THE SEED REPLICATE, and it is not optional. §41p's +1.70 pool-mean greedy is ~2sigma
#   against this line's measured +-0.87 seed noise, and its two pools DISAGREE IN SIGN on greedy (asdiv
#   -2.50 p=0.11, svamp +5.90 p=1.3e-4). This line has produced two retractions from single-seed reads.
#   `s47` is §41p rerun at seed 47 with nothing else changed, so the acc|ANSWERED gain is either
#   reproduced or withdrawn. ⚠️`--seed` reached TrainingArguments only from 255dff1 onward; opd_train.py
#   uses its own loop and seeds the data order directly, so the seed is live here.
#
# All three arms carry `closure_smoke.py` (200 greedy items, ~90 s, exit 3 above 45% unclosed) between
# training and the gate, because the cost of learning about a closure collapse from the gate is 40
# GPU-minutes and the cost of learning it from a smoke test is 90 seconds.
#
# ⚠️READ acc|ANSWERED. a4 has sat at ~50% against 3.5-think's 70% through thirteen arms while greedy
# moved on trace shape. Capability moved only if that number moves.
# =============================================================================
set -eo pipefail
REPO_ROOT="${REPO_ROOT:-/home/youzhi/ArgonneAI}"
cd "$REPO_ROOT"; mkdir -p report

module load python/miniforge-25.3.0
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_SHLVL
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# ⚠️HF_HUB_OFFLINE=1 is deliberate. A hub-id TEACHER is resolved from HF_HOME, and `export
# HF_HOME="${HF_HOME:-...}"` is a NO-OP when the environment already sets it -- as it does here
# (/project/rcc/youzhi/.cache/huggingface). If that cache happened not to hold the model, the job would try
# to DOWNLOAD 7.6 GB from a compute node, which either hangs or fails deep inside the run. Offline mode turns
# a cache miss into an immediate, legible error instead.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$REPO_ROOT/reasoning:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT=/project/rcc/youzhi/models/a4_think_final
A35=/project/rcc/youzhi/models/a35_effort/genfix46_a085
TEACHER="${TEACHER:-/project/rcc/youzhi/models/a35_effort/genfix46_a085}"
# ⚠️MUST match STUDENT. On-policy KD is only on-policy if the traces came from the checkpoint being trained;
# pointed at a stale dump it silently becomes off-policy imitation, which §41a measured as a NULL.
ROLL="${ROLL:-/project/rcc/youzhi/data/a4_dpo/a4_dpo_all.jsonl}"
STUDENT="${STUDENT:-$ROOT/think_combo}"
LR="${LR:-1e-5}"
ARMS="${ARMS:-notail anchor s47}"

for p in "$STUDENT/model.safetensors" "$ROLL"; do
  [ -e "$p" ] || { echo "FATAL missing $p"; exit 2; }
done
# ⚠️TEACHER may be a LOCAL PATH or a HUGGING FACE HUB ID (e.g. Qwen/Qwen3-4B, resolved from HF_HOME).
# Only a path can be stat'd. An earlier version checked "$TEACHER/config.json" unconditionally and killed
# the job instantly with `FATAL missing Qwen/Qwen3-4B/config.json` -- the SAME local-path assumption that had
# just been fixed inside opd_train.py, reproduced in the launcher's own pre-flight.
case "$TEACHER" in
  self) : ;;
  /*)   [ -e "$TEACHER/config.json" ] || { echo "FATAL missing $TEACHER/config.json"; exit 2; } ;;
  *)    echo "[preflight] TEACHER=$TEACHER treated as a HF hub id (HF_HOME=${HF_HOME:-default})" ;;
esac

# ⚠️Cheap consistency check: warn loudly if ROLL does not look like it came from STUDENT. This cannot be
# proven from filenames, but a mismatch between "the student is round 6" and "the dump is the original
# combo dump" is the single most likely way this job silently becomes off-policy imitation.
case "$ROLL" in
  *"$(basename "$STUDENT" | sed 's/^think_//')"*) : ;;
  *) echo "!! WARNING ROLL=$ROLL does not name STUDENT=$(basename "$STUDENT") -- if these are not the same"
     echo "!! policy, this run is OFF-policy imitation (measured a NULL, §41a), not on-policy KD." ;;
esac

echo "============================================================"
echo "a4 opd35 follow-up: protect termination + seed replicate   node=$(hostname) job=${SLURM_JOB_ID}"
echo "  student = $STUDENT"
echo "  teacher = $TEACHER   arms = $ARMS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

GATE=()
for ARM in $ARMS; do
  # ⚠️TAG exists because arm NAMES are not unique across jobs but `.opd_complete` is checked by PATH. Running
  # `ARMS=anchor` with a different TEACHER against the existing `think_opd35anchor` printed ">>> already
  # trained", trained nothing, and gated the OLD checkpoint under the new arm's label -- a confident wrong
  # answer with no error. Set TAG= whenever the teacher or student is not the default.
  OUT="$ROOT/think_opd35${TAG:-}$ARM"
  # ⚠️Bash loop variables are NOT scoped to the iteration. Without this reset, an arm that sets ALR leaks
  # its LR into every LATER arm that does not -- so `ARMS="repairlo repair"` would run both at 3e-6 and
  # report a flat dose-response that was never run. Reset before the case, not inside it.
  ALR=""; ATEACH=""
  case "$ARM" in
    notail) EXTRA="--kd-weight 1.0 --exclude-terminators 1 --ce-weight 0.0"; SEED=46 ;;
    anchor) EXTRA="--kd-weight 1.0 --exclude-terminators 0 --ce-weight 0.5"; SEED=46 ;;
    s47)    EXTRA="--kd-weight 1.0 --exclude-terminators 0 --ce-weight 0.0"; SEED=47 ;;
    # §41w pinned the mechanism: removing ALL gradient from the closure logits moved trace length by one
    # token, so the lengthening is BODY imitation of a longer-trace teacher, not a closure effect.
    # Prefix-only KD is the protection that matches that -- take the teacher's early decisions (where
    # §41b says the failure is: 79% of wrong traces differ at equation index 0) and leave the student's
    # own tail alone. pre50 keeps half the completion, pre33 a third.
    pre50)  EXTRA="--kd-weight 1.0 --exclude-terminators 0 --ce-weight 0.0 --kd-prefix-frac 0.5";  SEED=46 ;;
    pre33)  EXTRA="--kd-weight 1.0 --exclude-terminators 0 --ce-weight 0.0 --kd-prefix-frac 0.33"; SEED=46 ;;
    # and the combination, if prefix-only helps but not enough on its own
    pre50a) EXTRA="--kd-weight 1.0 --exclude-terminators 0 --ce-weight 0.5 --kd-prefix-frac 0.5";  SEED=46 ;;
    # §41y: ce-weight 0.5 recovered 39 of the 68 tokens the KD added (280 -> 241 vs the baseline's 212).
    # anchor1 asks whether the remaining 29 come back at twice the anchor weight, or whether the extra
    # likelihood pressure just undoes the capability gain -- the CE term pulls toward traces the model
    # was ALREADY trained on, so at some weight it must start erasing what the KD taught.
    anchor1) EXTRA="--kd-weight 1.0 --exclude-terminators 0 --ce-weight 1.0"; SEED=46 ;;
    s48)     EXTRA="--kd-weight 1.0 --exclude-terminators 0 --ce-weight 0.0"; SEED=48 ;;   # 3rd seed for the soup
    # §41ap/§41ar's REPAIR PASS: pure CE on the model's OWN verified-correct traces. No teacher (opd_train
    # skips loading one at --kd-weight 0), no new data, targets median 183 / p90 330 think tokens drawn from
    # the model's current style. It attacks the two monotone costs at once -- the 512-token unclosed TAIL and
    # the empty-think mode (a verified-correct trace is neither) -- and cannot pull toward a stale checkpoint
    # because the targets ARE the current checkpoint's output.
    # ⚠️Requires ROLL to point at the STUDENT's own rollout dump, not the original one.
    repair)  EXTRA="--kd-weight 0.0 --ce-weight 1.0 --labels correct"; SEED=46 ;;
    # ⚠️DOSE-RESPONSE, not a hedge. The repair pass has ONE way to fail and it is not "too weak": pure CE
    # on already-likely traces pulls the policy back toward what it already does, and six rounds of KD is
    # exactly what "what it already does" no longer is. §41y measured this trade at CE weight 0.5 MIXED
    # with KD -- it returned 39 of 68 added tokens while keeping 7.0 of 7.6pp of acc|ANSWERED -- but a pure
    # CE pass at weight 1.0 with no KD term holding the other side is stronger medicine, and the amount of
    # capability it erases is a function of the step size. Two LRs make "closure recovered per point of
    # acc|ANSWERED lost" a measured slope instead of one point with no error bar.
    repairlo) EXTRA="--kd-weight 0.0 --ce-weight 1.0 --labels correct"; SEED=46; ALR=3e-6 ;;
    # ⭐§41bg: BASE-MODEL TEACHERS + TERMINATOR MASKING. The audition measured both Qwen3 base models at a
    # teacher closure hazard of EXACTLY 0.0000 with p(</think>|closed)=0.00 -- `</think>` and `<|im_end|>`
    # are instruct-only control tokens a base model was never trained to emit, so it puts no mass on them
    # anywhere. Under reverse KL that is not a mild mismatch but a divergence: p_s log(p_s/p_t) blows up
    # wherever the student wants to close and the teacher assigns ~0, driving closure to zero HARDER than
    # any long-CoT teacher did. But the pathology is confined to TWO COLUMNS, and --exclude-terminators
    # removes exactly those two (151,669 -> 151,667), leaving revKL 0.75/0.82 at 78.7% agreement -- as much
    # signal as Qwen3-4B carries -- in the columns that remain.
    # ⚠️WHY §41w's NULL DOES NOT APPLY. --exclude-terminators was measured as a null with the 3.5-think
    # teacher, whose hazard already MATCHED the student's (ratio 1.00): masking a column the teacher was not
    # abusing changes nothing, correctly. That null was measured in the regime where the mask is
    # unnecessary. This is the regime where it is the entire point. A null does not generalise out of the
    # regime it was measured in.
    # ⚠️And a base model has no think-length POLICY to transfer -- conditioned on a4's own partial trace it
    # continues that trace's style, which is why the body-level lengthening that killed §41bb's Qwen3-4B
    # arms should not appear here. That is a PREDICTION; closure_smoke.py is the arbiter, not this comment.
    # q06base is the CONTROL: if a 0.6B teacher moves the gate as much as a 1.7B one, the gain is not
    # teacher capability and should not be attributed to it.
    q17base) EXTRA="--kd-weight 1.0 --exclude-terminators 1 --ce-weight 0.5"; SEED=46
             ATEACH=/project/rcc/youzhi/toxic-models/Qwen/Qwen3-1.7B-Base ;;
    q06base) EXTRA="--kd-weight 1.0 --exclude-terminators 1 --ce-weight 0.5"; SEED=46
             ATEACH=/project/rcc/youzhi/toxic-models/Qwen/Qwen3-0.6B-Base ;;
    *) echo "FATAL unknown arm $ARM"; exit 2 ;;
  esac
  ALR="${ALR:-$LR}"
  # per-arm teacher: q17base/q06base need DIFFERENT teachers within one job, which a job-level TEACHER
  # cannot express. Same reset-before-the-case discipline as ALR, for the same leak reason.
  ATEACH="${ATEACH:-$TEACHER}"
  # the job-level pre-flight above only validated $TEACHER; a per-arm one needs its own check, or a typo'd
  # path fails 4 minutes into the run instead of immediately
  case "$ATEACH" in
    self) : ;;
    /*)   [ -e "$ATEACH/config.json" ] || { echo "FATAL arm $ARM: missing $ATEACH/config.json"; exit 2; } ;;
  esac

  # ⚠️The repair arm trains on the STUDENT'S OWN rollouts. Pointed at the default dump it would train on a
  # checkpoint-from-four-rounds-ago's traces, which is precisely the "pull toward a stale checkpoint" the
  # arm exists to avoid -- and it would look like it ran correctly. Enforced, not commented.
  if [ "$ARM" = "repair" ] && [ "$ROLL" = "/project/rcc/youzhi/data/a4_dpo/a4_dpo_all.jsonl" ]; then
    echo "FATAL arm=repair needs ROLL= pointed at \$STUDENT's OWN rollout dump, not the default"; exit 2
  fi
  # ⚠️PROVENANCE GUARD. TAG prevents the collision when I remember to set it; this catches it when I don't.
  # A pre-existing checkpoint is only safe to REUSE if it was produced by the same student and teacher.
  # Markers written before this campaign hold a bare step count and carry no provenance -- those warn rather
  # than abort, because refusing them would block legitimate reuse of every older arm.
  if [ -f "$OUT/.opd_complete" ]; then
    python - "$OUT/.opd_complete" "$STUDENT" "$ATEACH" <<'PY'
import json, sys
marker, student, teacher = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(marker).read().split("\n", 1)
if len(txt) < 2 or not txt[1].strip():
    print(f"!! WARNING {marker} predates provenance recording (steps={txt[0].strip()}). Cannot verify it "
          f"came from this student/teacher; reusing it unchecked.")
    raise SystemExit(0)
p = json.loads(txt[1])
bad = []
if p.get("student") != student: bad.append(f"student: recorded {p.get('student')!r} != requested {student!r}")
if p.get("teacher") != teacher: bad.append(f"teacher: recorded {p.get('teacher')!r} != requested {teacher!r}")
if bad:
    print("FATAL this checkpoint was produced by a DIFFERENT run and would be gated under this arm's name:")
    for b in bad: print("   " + b)
    print("   -> set TAG= to a fresh prefix, or delete the directory to retrain.")
    raise SystemExit(3)
print(f"   provenance OK (steps={p['steps']}, teacher={p['teacher']})")
PY
    prc=$?
    [ $prc -eq 3 ] && exit 3
  fi
  if [ ! -f "$OUT/.opd_complete" ]; then
    echo; echo "######## $ARM : $EXTRA --lr $ALR --teacher $ATEACH ########"
    set +e
    python "$REPO_ROOT/reasoning/opd_train.py" \
      --student "$STUDENT" --model_def "$REPO_ROOT/model.py" --teacher "$ATEACH" \
      --div revkl $EXTRA \
      --rollouts "$ROLL" --out "$OUT" \
      --per-problem 3 --max-seq-len 1024 --rope-theta 1000000 \
      --lr "$ALR" --max-batch-tokens 8192 --grad-accum 1 --epochs 1 --warmup 20 \
      --teacher-temp 1.0 --grad-clip 1.0 \
      --seed "$SEED" --log-every 50 --stats-out "report/a4_opd35${ARM}_stats.json"
    rc=$?
    set -e
    echo "[$ARM train exit=$rc]"
    if [ $rc -ne 0 ]; then echo "!! $ARM did not train; skipping it"; continue; fi
  else
    echo ">>> $ARM already trained"
  fi

  echo; echo "######## $ARM : closure smoke test ########"
  set +e
  python "$REPO_ROOT/reasoning/closure_smoke.py" --model "$OUT" --pools asdiv --n 200 --warn-decoded 300 \
    --json-out "report/a4_opd35${ARM}_smoke.json"
  src=$?
  set -e
  if [ $src -eq 3 ]; then
    echo "!! $ARM FAILED the closure smoke test -- excluded from the gate"
  else
    [ $src -ne 0 ] && echo "!! smoke test could not run (exit $src); gating $ARM anyway, unchecked"
    GATE+=("a4opd35${ARM}_a100=$OUT")
  fi
done

if [ ${#GATE[@]} -eq 0 ]; then
  echo; echo "######## every arm failed the closure check; nothing to gate ########"
  echo "That would mean protecting termination cannot be done without breaking something else."
  exit 0
fi

echo; echo "######## GATE: ${GATE[*]} vs combo vs released 3.5-think ########"
# 3.5-think is deliberately NOT in this call. Its row is a fixed external reference and is already
# measured on identical items (same pool, same n, same seed) in every other gate JSON, so
# arms_table.py merges it in for free -- while including it here would cost ~20 GPU-minutes per stage
# for a paired p-value against a model nothing is being compared to pairwise. The comparisons that
# need pairing are a4-vs-a4, and both a4 references are in the call.
GATE+=("a4start_a100=$STUDENT" "a4combo_a100=$ROOT/think_combo")
set +e
python "$REPO_ROOT/reasoning/effort_gate.py" --models "${GATE[@]}" \
  --pools asdiv svamp --n 1000 --k 8 --json-out "report/a4kd3_gate_n1000.json"
echo "[gate asdiv/svamp exit=$?]"
python "$REPO_ROOT/reasoning/effort_gate.py" --models "${GATE[@]}" \
  --pools gsmplus mawps --n 500 --k 8 --json-out "report/a4kd3_gate_n500.json"
echo "[gate gsmplus/mawps exit=$?]"
python "$REPO_ROOT/reasoning/effort_gate.py" --models "${GATE[@]}" \
  --pools math500 --n 319 --k 8 --json-out "report/a4kd3_gate_math500.json"
echo "[gate math500 exit=$?]"
python "$REPO_ROOT/reasoning/gate_report.py" --json "report/a4kd3_gate_*.json" \
  --baseline a4combo_a100
set -e
echo; echo "######## DONE ########"
