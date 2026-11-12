Tinystory:
1.训练命令（PowerShell）
python -m basics.train `
  --data_dir ./data/tinystory `
  --device cuda `
  --vocab_size 50257 `
  --d_model 256 `
  --d_ff 768 `
  --context_len 256 `
  --num_heads 8 `
  --num_kv_heads 2 `
  --num_layers 4 `
  --rope_theta 10000 `
  --batch_size 8 `
  --max_lr 3e-4 `
  --min_lr 3e-5 `
  --warm_up_it 500 `
  --cosine_it 30000 `
  --train_steps 30000 `
  --val_interval 500 `
  --val_batches 20 `
  --save_interval 2500 `
  --log_interval 1 `
  --save_ckp_path ./checkpoints/tinystories `
  --no_wandb
2.文本生成
python -m basics.generate `
  --checkpoint ./checkpoints/tinystories/final_checkpoint_30000.pt `
  --prompt "There were three story of the three body porblem questions" `
  --max_tokens 50 `
  --temperature 0.8 `
  --top_p 0.9 `
  --n_samples 3 `
  --device cuda

OWT:
1.训练命令（PowerShell）
python -m basics.train `
  --data_dir ./data/openwebtext `
  --device cuda `
  --vocab_size 50257 `
  --d_model 256 `
  --d_ff 768 `
  --context_len 256 `
  --num_heads 8 `
  --num_kv_heads 2 `
  --num_layers 4 `
  --rope_theta 10000 `
  --batch_size 8 `
  --max_lr 3e-4 `
  --min_lr 3e-5 `
  --warm_up_it 500 `
  --cosine_it 30000 `
  --train_steps 30000 `
  --weight_decay 0.01 `
  --beta1 0.9 `
  --beta2 0.95 `
  --clip_grad_norm 1.0 `
  --val_interval 500 `
  --val_batches 20 `
  --save_interval 2500 `
  --log_interval 1 `
  --save_ckp_path ./checkpoints/owt `
  --wandb_project mini-gpt `
  --wandb_run_name owt-gqa-8q2kv-30k
2.文本生成
python -m basics.generate `
  --checkpoint ./checkpoints/owt/final_checkpoint_30000.pt `
  --prompt "The development of artificial intelligence" `
  --max_tokens 50 `
  --temperature 0.8 `
  --top_p 0.9 `
  --n_samples 3 `
  --device cuda