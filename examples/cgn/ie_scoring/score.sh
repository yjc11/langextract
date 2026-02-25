# python3 score_v2.py \
#     --pred_file /opt/ie_benchmark/llm_uie_sft/output/gpt-4-turbo-2024-04-09_result.jsonl \
#     --val_file /opt/ie_benchmark/llm_uie_sft/datasets/long_short_v2_doc_8000_one_shot_n_ratio0/val.json \
#     --save_dir /opt/ie_benchmark/llm_uie_sft/metrics/meteor \
#     --metric meteor \
#     --exclude_keys "" 

# python3 score_v2.py \
#     --pred_file /opt/ie_benchmark/llm_uie_sft/output/gpt-4-turbo-2024-04-09_result.jsonl \
#     --val_file /opt/ie_benchmark/llm_uie_sft/datasets/long_short_v2_doc_8000_one_shot_n_ratio0/val.json \
#     --save_dir /opt/ie_benchmark/llm_uie_sft/metrics/bleu \
#     --metric bleu-4\
#     --exclude_keys "" 

# python3 score_v2.py \
#     --pred_file /opt/ie_benchmark/llm_uie_sft/output/gpt-4-turbo-2024-04-09_result.jsonl \
#     --val_file /opt/ie_benchmark/llm_uie_sft/datasets/long_short_v2_doc_8000_one_shot_n_ratio0/val.json \
#     --save_dir /opt/ie_benchmark/llm_uie_sft/metrics/rouge-l \
#     --metric rouge-l\
#     --exclude_keys "" 

# python3 score_v2.py \
#     --pred_file /opt/ie_benchmark/llm_uie_sft/output/gpt-4-turbo-2024-04-09_result.jsonl \
#     --val_file /opt/ie_benchmark/llm_uie_sft/datasets/long_short_v2_doc_8000_one_shot_n_ratio0/val.json \
#     --save_dir /opt/ie_benchmark/llm_uie_sft/metrics/rocr \
#     --metric rocr \
#     --exclude_keys "" 

python3 score_v2.py \
    --pred_file /opt/ie_benchmark/llm_uie_sft/datasets/long_short_v2_doc_8000_default_n_ratio0/full.jsonl \
    --val_file /opt/ie_benchmark/llm_uie_sft/datasets/long_short_v2_doc_8000_one_shot_n_ratio0/val.json \
    --save_dir /opt/ie_benchmark/llm_uie_sft/metrics/meteor \
    --metric meteor \
    --exclude_keys "" 