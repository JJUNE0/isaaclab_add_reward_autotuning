# TITA CO-RL/MOOPPO evaluation

These evaluators implement the TITA MOO policy contracts directly:

- observation: 32D (`stack_policy` 28D + velocity command 4D)
- policy action order: `L1,R1,L2,R2,L3,R3,L4,R4`
- leg action: absolute position target with scale `1.0`
- wheel action: velocity target with scale `40.0`

Run Isaac behavior evaluation (also exports a checkpoint-specific ONNX):

```bash
python eval/eval_behavior.py \
  --checkpoint /absolute/path/to/model_1999.pt \
  --num_envs 8 --headless
```

Run headless MuJoCo Sim2Sim evaluation:

```bash
conda run -n cosim python eval/eval_sim2sim.py \
  --onnx eval/results/model_1999.onnx
```

The Sim2Sim evaluator resets at yaw 180 degrees by default so the robot faces
away from the stairs. Override this with `--reset_yaw_deg`.

To watch the same evaluation in real time:

```bash
conda run -n cosim python eval/eval_sim2sim.py \
  --onnx eval/results/model_1999.onnx --viewer
```

Measure zero-command drift for 30 seconds:

```bash
conda run -n cosim python eval/eval_sim2sim_zero_command.py \
  --onnx eval/results/model_1999.onnx \
  --duration_s 30
```

By default, Isaac behavior evaluation disables observation noise, DomainManager,
pushes, randomized reset poses, randomized joint resets, and randomized material
events so checkpoint comparisons use nominal deterministic conditions. Pass
`--keep_randomization` to retain the Play task randomization.
