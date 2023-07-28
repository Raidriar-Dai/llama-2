# Inference with Fine-tuned Chat Models
torchrun --nproc_per_node 8 --master_port 12345 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 4 