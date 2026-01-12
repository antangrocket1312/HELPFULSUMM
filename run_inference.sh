torchrun --nproc_per_node=1 --master_port=9778 inference/inference.py \
    --policy_base_model hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 \
    --policy_model_path ./models/stage_2_helpfulsumm_rl/step-200 \
    --inference_data_path ./data/test/test.pkl \
    --summary_output_dir ./output/stage_2_rl_inference/summary \
    --kp_extraction_output_dir ./output/stage_2_rl_inference/summary_kp_extraction \
    --openai_api_key $OPENAI_API_KEY \
    --seed 201