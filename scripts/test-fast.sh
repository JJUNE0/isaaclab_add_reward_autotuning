#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "ERROR: Not inside a Git repository."
    exit 1
}

cd "$ROOT"

echo "[TEST-FAST] Checking Python Syntax across lab and sim2sim..."
python3 -c "
import py_compile, glob

files = glob.glob('lab/**/*.py', recursive=True) + glob.glob('sim2sim/**/*.py', recursive=True) + glob.glob('test/**/*.py', recursive=True)
passed = 0
failed = 0
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        passed += 1
    except Exception as e:
        print(f'FAIL: {f} -> {e}')
        failed += 1

print(f'[TEST-FAST] Syntax Check Results: {passed} passed, {failed} failed.')
if failed > 0:
    exit(1)
"

echo "[TEST-FAST] Checking XML / URDF files..."
python3 -c "
import xml.etree.ElementTree as ET, glob

urdfs = glob.glob('sim2sim/**/*.urdf', recursive=True)
for u in urdfs:
    try:
        ET.parse(u)
    except Exception as e:
        print(f'FAIL URDF XML Parse: {u} -> {e}')
        exit(1)
print(f'[TEST-FAST] XML Parse Check: {len(urdfs)} URDF files OK.')
"

echo "[TEST-FAST] All fast tests passed."
