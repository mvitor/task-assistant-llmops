import json
import sys
import os

BASELINE_FILE = "src/task_assistant/monitoring/baseline_scores.json"
CURRENT_FILE = "src/task_assistant/monitoring/current_scores.json"

REGRESSION_THRESHOLD = 0.15  # 15% tolerance — accounts for LLM output variance

if not os.path.exists(BASELINE_FILE):
    print("⚠️  No baseline found. Creating baseline from current scores.")
    with open(CURRENT_FILE) as f:
        current = json.load(f)
    with open(BASELINE_FILE, "w") as f:
        json.dump(current, f, indent=2)
    print(f"✅ Baseline created: {current}")
    sys.exit(0)

with open(BASELINE_FILE) as f:
    baseline = json.load(f)

with open(CURRENT_FILE) as f:
    current = json.load(f)

failed = False

for metric in ["factual_accuracy_mean", "task_assistant_quality_mean"]:
    baseline_score = baseline[metric]
    current_score = current[metric]

    if current_score < baseline_score - REGRESSION_THRESHOLD:
        print(
            f"❌ Regression detected in {metric}: "
            f"{current_score:.3f} vs baseline {baseline_score:.3f}"
        )
        failed = True
    else:
        print(
            f"✅ {metric} maintained: "
            f"{current_score:.3f} vs baseline {baseline_score:.3f}"
        )

if failed:
    print("\n❌ Quality gates FAILED")
    sys.exit(1)
else:
    print("\n✅ All quality gates passed")
    sys.exit(0)
