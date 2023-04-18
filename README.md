# llm-dvlab
Training large language mode for DVLab

## Example
A fast example to train [alpaca](https://github.com/tatsu-lab/stanford_alpaca) on V100 / 3090ti.

### 1. Download stanford_alpaca codebase 
```
git clone https://github.com/tatsu-lab/stanford_alpaca`
cd stanford_alpaca
pip install -r requirements.txt
```
### 2. Convert llama checkpoints to huggingface version
```
git clone https://github.com/huggingface/transformers.git`
cd transformers
git checkout 165dd6dc916a43ed9b6ce8c1ed62c3fe8c28b6ef
pip install -e .
```

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
      --input_dir /models/llama \
      --model_size 7B \
      --output_dir /models/llama_converted
```
### 3. Install deepspeed
`pip install deepspeed`
download the [deepspeed config file](ds_config.json).

### 4. Training
```
torchrun --nproc_per_node=4 --master_port=6006 train.py \
	--model_name_or_path /data/llama_hf/7B \
	--data_path ./alpaca_data.json \
	--bf16 False \
	--output_dir /data/models/llama-7b \
	--num_train_epochs 3 \
	--per_device_train_batch_size 3 \
	--per_device_eval_batch_size 3 \
	--gradient_accumulation_steps 5 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 2000 \
	--save_total_limit 1 \
	--learning_rate 2e-5 \
	--weight_decay 0. \
	--warmup_ratio 0.03 \
	--deepspeed "ds_config.json" \
	--fp16 \
	--tf32 False
```
Note that you can change the batch size from 3 to 2, if you meet out-of-memory on 3090ti.
