# Code publication validation

- Execution Domain: DESKTOP_DRY_RUN
- Policy Checkpoint SHA256 / ONNX Hash: N/A (no policy execution or export)
- Evidence Status: PASS (static checks only); NOT_TESTED (training and simulation)
- Scope: tracked training changes, TITA task/configuration, estimator and multi-head RMA modules, evaluation tools, tests, and check scripts.
- Observed behavior: agent-check.sh passed: 321 Python files compiled, 0 failures; XML check found 0 URDF files. Selected Python sources also passed AST parsing, including scripts/co_rl and eval which test-fast.sh omits.
- Preparation: repaired an accidental URL fragment in the Flamingo configuration base-class name; removed trailing whitespace from new TITA configurations and AGENTS.md.
- Excluded: .omc state, existing experiment outputs, eval/results, binary robot assets, personal notes, and the blanket-staging agent-publish.sh helper.
- Limitations: no runtime/unit-test execution; TITA URDF and some evaluation defaults reference external local paths. Existing evaluation tools retain their output conventions. No hardware operations.
