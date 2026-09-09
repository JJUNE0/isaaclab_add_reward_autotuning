#!/usr/bin/env bash

set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "ERROR: Not inside a Git repository."
    exit 1
}

cd "$ROOT"

result=0

echo
echo "=================================================="
echo "Branch"
echo "=================================================="
git branch --show-current

echo
echo "=================================================="
echo "Git status"
echo "=================================================="
git status --short

echo
echo "=================================================="
echo "Changed files"
echo "=================================================="
git diff HEAD --name-status

echo
echo "=================================================="
echo "Diff statistics"
echo "=================================================="
git diff HEAD --stat

echo
echo "=================================================="
echo "Diff integrity check"
echo "=================================================="
if ! git diff HEAD --check; then
    echo "ERROR: git diff --check failed."
    result=1
else
    echo "PASS: No whitespace or conflict-marker errors."
fi

echo
echo "=================================================="
echo "Fast tests"
echo "=================================================="
if [[ -x "./scripts/test-fast.sh" ]]; then
    if ! ./scripts/test-fast.sh; then
        echo "ERROR: Fast tests failed."
        result=1
    fi
else
    echo "SKIP: scripts/test-fast.sh does not exist or is not executable."
fi

echo
echo "=================================================="
echo "Experiment Log Directory Check"
echo "=================================================="
if [[ -d "docs/exp" ]]; then
    invalid_exp_dirs=$(find docs/exp -mindepth 1 -maxdepth 1 -type d ! -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]" 2>/dev/null || true)
    if [[ -n "$invalid_exp_dirs" ]]; then
        echo "ERROR: Invalid experiment directory format found in docs/exp/:"
        echo "$invalid_exp_dirs"
        echo "Directory names in docs/exp/ must follow YYYY-MM-DD format (no hours/minutes)."
        result=1
    else
        echo "PASS: docs/exp/ directory naming convention OK."
    fi
fi

echo
echo "=================================================="
echo "Experiment Log Existence Check"
echo "=================================================="
# Check if code files were modified
code_changed=$(git diff HEAD --name-only 2>/dev/null | grep -E "^(lab/|sim2sim/|sdk/|tita_gluon_ws/)" || true)
exp_logged=$(git diff HEAD --name-only 2>/dev/null | grep -E "^docs/exp/.*\.md$" || true)
untracked_exp=$(git status --porcelain 2>/dev/null | grep -E "^\?\? docs/exp/.*\.md$" || true)

if [[ -n "$code_changed" ]]; then
    if [[ -z "$exp_logged" && -z "$untracked_exp" ]]; then
        echo "NOTICE: Code changes detected in task, but no new experiment log found in docs/exp/YYYY-MM-DD/*.md."
        echo " -> Reminder: If this change includes an experiment or test run, please create a report using docs/exp/template.md."
    else
        echo "PASS: Code changes accompanied by experiment log entry in docs/exp/."
    fi
else
    echo "PASS: No core code changes, experiment log check bypassed."
fi



echo
echo "=================================================="
echo "Final result"
echo "=================================================="

if [[ "$result" -eq 0 ]]; then
    echo "PASS"
else
    echo "FAIL"
fi

exit "$result"
