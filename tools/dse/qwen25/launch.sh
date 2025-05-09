export HF_HOME=/media/baie_nas/data/datasets/hf/dse
export PYTHONPATH=./:$PYTHONPATH
python train.py \
       --output_dir /data1/infantes/dse_qwen25 \
       --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
       --lora \
       --save_steps 100 \
       --dataset_name Tevatron/wiki-ss-nq \
       --corpus_name Tevatron/wiki-ss-corpus \
       --bf16 True \
       --pooling eos \
       --tf32 True \
       --normalize True \
       --temperature 0.02 \
       --per_device_train_batch_size 2 \
       --gradient_checkpointing True \
       --train_group_size 2 \
       --learning_rate 1e-5 \
       --query_max_len 128 \
       --passage_max_len 4096 \
       --num_train_epochs 1 \
       --logging_steps 1 \
       --overwrite_output_dir True \
       --gradient_accumulation_steps 16 \
       --dataset_cache_dir /media/baie_nas/data/datasets/hf/dse
