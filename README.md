Activate venv:
source .venv/bin/activate

Dependencies:
python -m pip install --upgrade pip
python -m pip install torch numpy

Train and evaluate:
Apple Silicon: python -m grammar_gpt.run_experiment --make_data --device mps --seed 123
CPU: python -m grammar_gpt.run_experiment --make_data --device cpu --seed 123
CUDA: python -m grammar_gpt.run_experiment --make_data --device cuda --seed 123

View results:
cat runs/eval_A/summary.json
cat runs/eval_B/summary.json
cat runs/eval_C/summary.json

Training logs:
head -n 20 runs/train_A/log.jsonl
tail -n 20 runs/train_A/log.jsonl

Re-run evaluation only (no data rebuild):
python -m grammar_gpt.run_experiment --device mps --seed 123 (Dropped -make date)

Re-run from scratch and start clean:
rm -rf runs
python -m grammar_gpt.run_experiment --make_data --device mps --seed 123
