# llama-2: Inference with Fine-tuned Chat Models
# torchrun --nproc_per_node 8 --master_port 12345 example_chat_completion.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 4 



# Obtain logdifference results with Baichuan-13B-Chat
python logdifference_calculation.py --dataset="gsm8k" --model="baichuan-inc/Baichuan-13B-Chat" --qes_limit=0 \
    --prompt_path="./validation_prompts/math_word_problems" --random_seed=42 --output_dir="./logdifference_results"