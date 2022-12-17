
export CUDA_VISIBLE_DEVICES=1

XLA_PYTHON_CLIENT_PREALLOCATE=false python -um batch_rl.fixed_replay.train \
  --base_dir=/tmp/batch_rl \
  --replay_dir=$DATA_DIR/Breakout/1 \
  --agent_name=jax_dqn \
  --gin_files='batch_rl/fixed_replay/configs/jax_dqn.gin'