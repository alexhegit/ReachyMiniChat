#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/venv/bin/python"

JSON_MODE=0
JSON_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_MODE=1
      shift
      ;;
    --json-out)
      JSON_MODE=1
      JSON_OUT="${2:-}"
      if [[ -z "$JSON_OUT" ]]; then
        echo "❌ --json-out requires a file path"
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/ci_smoke.sh [--json] [--json-out FILE]

Options:
  --json             Print machine-readable JSON summary at the end.
  --json-out FILE    Save JSON summary to FILE (also enables --json).
  -h, --help         Show this help.
EOF
      exit 0
      ;;
    *)
      echo "❌ Unknown option: $1"
      echo "Try: ./scripts/ci_smoke.sh --help"
      exit 2
      ;;
  esac
done

if [[ ! -x "$PY" ]]; then
  echo "❌ Python venv not found: $PY"
  echo "Run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
  exit 2
fi

echo "== ReachyCheese CI Smoke =="
echo "Project: $ROOT_DIR"
echo "Python : $PY"
echo

fail=0

TEST_NAMES=()
TEST_CMDS=()
TEST_RCS=()
TEST_LOGS=()
REPORT_ROWS="$(mktemp)"

run_test() {
  local name="$1"
  local cmd="$2"
  local log_file
  log_file="$(mktemp)"

  TEST_NAMES+=("$name")
  TEST_CMDS+=("$cmd")
  TEST_LOGS+=("$log_file")

  echo "--- [START] $name ---"
  # Capture and mirror output for both human logs and optional JSON report.
  bash -lc "$cmd" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  TEST_RCS+=("$rc")

  printf '%s\t%s\t%s\t%s\n' "$name" "$cmd" "$rc" "$log_file" >> "$REPORT_ROWS"

  if [[ $rc -eq 0 ]]; then
    echo "--- [PASS ] $name ---"
  else
    echo "--- [FAIL ] $name (exit $rc) ---"
    fail=1
  fi
  echo
}

run_test "E2E smoke (daemon/camera/motion/photo)" "cd '$ROOT_DIR' && '$PY' utils/auto_smoke_reachycheese.py"
run_test "Face aligner simulation logic" "cd '$ROOT_DIR' && '$PY' utils/test_reachycheese_aligner_sim.py"

if [[ $fail -eq 0 ]]; then
  echo "✅ All smoke tests passed."
else
  echo "❌ One or more smoke tests failed."
fi

if [[ $JSON_MODE -eq 1 ]]; then
  # Export context for JSON generator.
  export ROOT_DIR PY REPORT_ROWS

  # Build JSON via Python for robust escaping.
  REPORT_JSON="$($PY - <<'PY'
import csv
import json
import os
from datetime import datetime, timezone

rows_path = os.environ.get('REPORT_ROWS', '')
results = []
overall_pass = True

if rows_path and os.path.exists(rows_path):
    with open(rows_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 4:
                continue
            name, cmd, rc_raw, log_path = row[0], row[1], row[2], row[3]
            try:
                rc = int(rc_raw)
            except Exception:
                rc = 999

            log_tail = ''
            if log_path and os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as lf:
                        content = lf.read()
                    log_tail = content[-3000:]
                except Exception:
                    log_tail = ''

            passed = (rc == 0)
            overall_pass = overall_pass and passed
            results.append({
                'name': name,
                'command': cmd,
                'exit_code': rc,
                'passed': passed,
                'log_tail': log_tail,
            })

report = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'project_root': os.environ.get('ROOT_DIR', ''),
    'python': os.environ.get('PY', ''),
    'overall_passed': overall_pass,
    'tests': results,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
)"

  echo
  echo "== JSON Report =="
  echo "$REPORT_JSON"

  if [[ -n "$JSON_OUT" ]]; then
    mkdir -p "$(dirname "$JSON_OUT")"
    printf '%s\n' "$REPORT_JSON" > "$JSON_OUT"
    echo "📄 JSON written to: $JSON_OUT"
  fi
fi

# Cleanup temp logs
for logf in "${TEST_LOGS[@]}"; do
  [[ -f "$logf" ]] && rm -f "$logf"
done
[[ -f "$REPORT_ROWS" ]] && rm -f "$REPORT_ROWS"

exit $([[ $fail -eq 0 ]] && echo 0 || echo 1)
