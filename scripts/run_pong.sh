

export CUDA_VISIBLE_DEVICES=1

# export XLA_PYTHON_CLIENT_PREALLOCATE=false 
python -um batch_rl.fixed_replay.train \
  --base_dir=/raid/mitsuhiko/batch_rl/Pong \
  --replay_dir=$DATA_DIR/Pong/1 \
  --agent_name=jax_dqn \
  --gin_files='batch_rl/fixed_replay/configs/jax_dqn.gin' \
  --gin_bindings='atari_lib.create_atari_environment.game_name = "Pong"'