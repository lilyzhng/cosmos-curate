# Qwen 2.5 VL Finetuning with Cosmos-Curate

## Objective
Use Cosmos-Curate to create training datasets for finetuning Qwen 2.5 VL with improved spatial-temporal understanding.

## Prerequisites
- Raw video datasets
- Cosmos-Curate environment setup
- Qwen 2.5 VL model access

## Step 1: Generate Training Data

### Run Cosmos-Curate Pipeline
```bash
cosmos-curate video split \
  --input-videos /path/to/videos \
  --output-prefix s3://bucket/qwen-training \
  --embedding-algorithm internvideo2 \
  --prompt-types default \
  --generate-t5-embeddings \
  --window-size 256 \
  --sampling-fps 2.0 \
  --motion-filter-threshold 0.01 \
  --aesthetic-filter-threshold 0.5
```

### What This Generates
- **Video clips**: `{clip-uuid}.mp4` (shot-based segments)
- **Training windows**: `{clip-uuid}_{0_255}.mp4` (256-frame windows)
- **Captions**: `v0/all_window_captions.json` (Qwen-generated descriptions)
- **Embeddings**: `{clip-chunk-uuid}.parquet` (InternVideo2 features)
- **Metadata**: `{clip-uuid}.json` (motion/aesthetic scores)

## Step 2: Prepare Qwen Training Format

### Extract Training Pairs
```python
# Convert Cosmos-Curate output to Qwen training format
import json
import os

def create_qwen_training_data(cosmos_output_dir):
    # Load captions
    with open(f"{cosmos_output_dir}/v0/all_window_captions.json") as f:
        captions = json.load(f)
    
    training_data = []
    for window_id, caption_data in captions.items():
        video_path = f"clips/{window_id}.mp4"
        if os.path.exists(f"{cosmos_output_dir}/{video_path}"):
            training_data.append({
                "video": video_path,
                "conversations": [{
                    "from": "human", 
                    "value": "Describe what happens in this video."
                }, {
                    "from": "gpt",
                    "value": caption_data["qwen_caption"]
                }]
            })
    
    return training_data
```

## Step 3: Finetune Qwen 2.5 VL

### Training Configuration
```python
# qwen_training_config.py
training_args = {
    "model_name": "Qwen/Qwen2-VL-2B-Instruct",
    "data_path": "qwen_training_data.json",
    "output_dir": "./qwen-spatial-temporal",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-5,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "save_steps": 500,
    "save_total_limit": 1,
    "max_seq_length": 2048,
    "bf16": True,
    "tf32": True,
    "dataloader_num_workers": 4,
    "gradient_checkpointing": True,
    "report_to": "none"
}
```

### Launch Training
```bash
# Install Qwen training dependencies
pip install transformers accelerate deepspeed

# Run finetuning
python qwen_finetune.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --data_path qwen_training_data.json \
  --output_dir ./qwen-spatial-temporal \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-5
```

## Step 4: Evaluate Results

### Test Spatial-Temporal Understanding
```python
# test_model.py
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Load finetuned model
model = Qwen2VLForConditionalGeneration.from_pretrained("./qwen-spatial-temporal")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Test on new videos
def test_spatial_temporal(video_path):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": "Describe the spatial relationships and temporal changes in this video."}
        ]
    }]
    
    inputs = processor.apply_chat_template(messages, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(output[0], skip_special_tokens=True)
```

## Expected Improvements

### Before Finetuning
- Generic video descriptions
- Limited spatial awareness
- Poor temporal understanding

### After Finetuning
- Detailed spatial relationships ("object A moves from left to right")
- Temporal dynamics ("acceleration occurs over 3 seconds")
- Motion patterns ("smooth vs jerky movement")

## Success Metrics

1. **Spatial Accuracy**: Correctly identify object positions and relationships
2. **Temporal Precision**: Accurately describe timing and motion patterns  
3. **Caption Quality**: More detailed and accurate descriptions
4. **Consistency**: Stable performance across different video types

## Next Steps

1. **Run the pipeline** on your video dataset
2. **Convert outputs** to Qwen training format
3. **Finetune model** with spatial-temporal focus
4. **Evaluate performance** on held-out test set
5. **Iterate** with additional data or training adjustments