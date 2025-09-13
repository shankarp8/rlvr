set -x


export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs"
# export BASE_MODEL="/home/sp2583/rlvr/Qwen3-4B-Thinking"
# export BASE_MODEL='/home/sp2583/rlvr/distill_qwen_1.5b'
export BASE_MODEL='/home/sp2583/rlvr/Qwen2.5-3B-Instruct'
# export BASE_MODEL='/home/sp2583/rlvr/outputs/confidence_after_answer_plausible/qwen3_trylongbasic_1e-6/global_step_200/actor'

N_GPUS=1
ROLLOUT_N=10
MAX_LENGTH=2048
TENSOR_MODEL_PARALLEL_SIZE=1
TOTAL_EPOCHS=1
SAVE_STEPS=50
EVAL_STEPS=5

LR=3e-6

EXPERIMENT_NAME="qwen3_3e-6"
PROJECT_NAME='confidence_after_answer_plausible'


python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 +algorithm.std_norm=False \
 data.train_files=$HOME/rlvr/rlcr_pqa_train.parquet \
 data.val_files=$HOME/rlvr/rlcr_pqa_validation.parquet \
 data.train_batch_size=32 \
 data.val_batch_size=256 \
 data.max_prompt_length=3072 \
 data.max_response_length=$MAX_LENGTH \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=$LR \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.kl_loss_coef=0.005 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=1.0 \
 +actor_rollout_ref.rollout.val_temperature=0.5 \
 actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 +actor_rollout_ref.rollout.n_val=1 \
 trainer.critic_warmup=0 \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.checkpoints_dir=$CHECKPOINTS_DIR \
 trainer.resume_mode='disable' \
 trainer.logger=['console','wandb'] \
 trainer.val_generations_to_log_to_wandb=10 \
 +trainer.val_before_train=False \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=$SAVE_STEPS \
 trainer.test_freq=$EVAL_STEPS \
 trainer.total_epochs=$TOTAL_EPOCHS \
 +trainer.vary_confidence=False \
 +trainer.parallel_confidence=True \
 +trainer.num_duplicated_rollouts=1

