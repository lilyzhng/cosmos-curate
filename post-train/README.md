# Qwen 2.5 VL Finetuning with Cosmos-Curate

## Objective
Use Cosmos-Curate to create training datasets for finetuning Qwen 2.5 VL with improved spatial-temporal understanding using Cosmos-Reason1's proven training strategy.

## Prerequisites
- Raw video datasets
- Cosmos-Curate environment setup
- Qwen 2.5 VL model access

## Training Strategy Comparison

| Aspect | Cosmos-Reason1 | ALFA-VL (Our Approach) |
|--------|----------------|-------------------------|
| **Base Model** | Custom hybrid Mamba-MLP-Transformer | Qwen 2.5 VL (Pre-trained) |
| **Model Sizes** | 8B, 56B parameters | 2B, 7B parameters |
| **Stage 1: Vision Pre-training** | 120M image/video samples | ❌ Skip (Qwen has this) |
| **Stage 2: General SFT** | 8M diverse vision-language | ❌ Skip (Qwen has this) |
| **Stage 3: Physical AI SFT** | Custom physics reasoning data | ✅ **Cosmos-Curate + Chain-of-Thought** |
| **Stage 4: Physical AI RL** | PPO for reasoning chains | 🔄 Optional (DPO/PPO) |
| **Data Source** | Curated physics datasets | **Cosmos-Curate pipeline** |
| **Key Innovation** | Hybrid architecture + RL | **Real-world video + Metadata** |
| **Training Focus** | General physical reasoning | **Spatial-temporal understanding** |
| **Reasoning Format** | Long chain-of-thought | **Structured analysis (Spatial→Temporal→Physics)** |
| **Data Scale** | Large-scale custom curation | **Efficient targeted curation** |
| **Evaluation** | Physics benchmarks | **Real-world video understanding** |

## Our ALFA-VL Strategy (Adapted from Cosmos-Reason1)

Following the [Cosmos-Reason1 paper](https://arxiv.org/html/2503.15558v1), we'll adapt their **4-stage training approach** for Qwen 2.5 VL:

1. **Vision Pre-training**: ❌ *Skip - Qwen already has this*
2. **General SFT**: ❌ *Skip - Qwen already has this*  
3. **Physical AI SFT**: ✅ **Our main focus** - Cosmos-Curate + Chain-of-Thought
4. **Physical AI RL**: 🔄 *Optional advanced step*

### ALFA-VL Key Advantages:
- **Real-world video data** from Cosmos-Curate vs synthetic physics data
- **Rich metadata** (motion, aesthetics, embeddings) for enhanced reasoning
- **Efficient approach** - leverage pre-trained Qwen vs training from scratch
- **Structured reasoning** - Spatial → Temporal → Physics analysis format

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

## Step 2: Create Physical AI Training Data (Cosmos-Reason1 Style)

### Stage 1: Basic Training Data
```python
# Convert Cosmos-Curate output to basic training format
import json
import os
import glob

def create_basic_training_data(cosmos_output_dir):
    training_data = []
    with open(f"{cosmos_output_dir}/v0/all_window_captions.json") as f:
        captions = json.load(f)
    
    for window_id, caption_data in captions.items():
        training_data.append({
            "video": f"clips/{window_id}.mp4",
            "conversations": [{
                "from": "human", 
                "value": "Describe what you see in this video."
            }, {
                "from": "gpt",
                "value": caption_data["qwen_caption"]
            }]
        })
    return training_data
```

### Stage 2: Physical AI SFT Data (Key Innovation)
```python
# Create spatial-temporal reasoning data with chain-of-thought
def create_physical_ai_sft_data(cosmos_output_dir):
    training_data = []
    
    # Load motion and aesthetic metadata
    metadata_files = glob.glob(f"{cosmos_output_dir}/clips/*.json")
    
    for metadata_file in metadata_files:
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Create reasoning chains for spatial-temporal understanding
        if metadata.get("motion_score", {}).get("global_mean", 0) > 0.02:
            motion_score = metadata["motion_score"]["global_mean"]
            aesthetic_score = metadata.get("aesthetic_score", 0)
            
            training_data.append({
                "video": f"clips/{metadata['span_uuid']}.mp4",
                "conversations": [{
                    "from": "human",
                    "value": "Analyze the spatial relationships and temporal dynamics in this video. Think step by step about the physics involved."
                }, {
                    "from": "gpt", 
                    "value": f"""Let me analyze this video step by step:

**Spatial Analysis:**
- Objects are positioned across the frame with clear depth relationships
- Motion score of {motion_score:.3f} indicates {'high' if motion_score > 0.05 else 'moderate'} movement activity
- Aesthetic score of {aesthetic_score:.2f} suggests {'good' if aesthetic_score > 0.6 else 'average'} visual composition
- Spatial relationships show realistic positioning and scale

**Temporal Analysis:**  
- Movement patterns demonstrate {'rapid' if motion_score > 0.05 else 'gradual'} changes over the 8.5-second window
- The motion appears {'smooth and continuous' if motion_score < 0.1 else 'dynamic with quick transitions'}
- Temporal consistency follows expected real-world physics

**Physical Reasoning:**
Based on the motion patterns and spatial layout, this represents typical real-world physics where objects follow expected trajectories, maintain proper scale relationships, and exhibit realistic interaction dynamics. The movement patterns are consistent with natural forces and constraints."""
                }]
            })
    
    return training_data
```

## Step 3: Multi-Stage Finetuning (Cosmos-Reason1 Approach)

### Stage 1: Basic SFT (Optional Warmup)
```python
# Stage 1: Basic capabilities (optional - Qwen already has good general capabilities)
basic_config = {
    "model_name": "Qwen/Qwen2-VL-2B-Instruct",
    "data_path": "basic_training_data.json", 
    "output_dir": "./qwen-basic-sft",
    "num_train_epochs": 1,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 4,
    "max_seq_length": 1024
}
```

### Stage 2: Physical AI SFT (Main Training)
```python
# Stage 2: Spatial-temporal reasoning focus (MOST IMPORTANT)
physical_ai_config = {
    "model_name": "Qwen/Qwen2-VL-2B-Instruct",  # or "./qwen-basic-sft" if using Stage 1
    "data_path": "physical_ai_sft_data.json",
    "output_dir": "./qwen-physical-ai-sft", 
    "num_train_epochs": 3,
    "learning_rate": 1e-5,  # Lower LR for reasoning tasks
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 2048,  # Longer for reasoning chains
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 10,
    "save_steps": 500,
    "evaluation_strategy": "steps",
    "eval_steps": 500
}
```

### Launch Physical AI Training
```bash
# Install dependencies
pip install transformers accelerate deepspeed trl

# Run Physical AI SFT (Main Training)
python train_physical_ai.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --data_path physical_ai_sft_data.json \
  --output_dir ./qwen-physical-ai-sft \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 8 \
  --warmup_ratio 0.03 \
  --max_seq_length 2048 \
  --bf16 \
  --gradient_checkpointing
```

### Stage 3: Physical AI RL (Advanced - Optional)
```python
# Stage 3: Reinforcement learning for reasoning chains (Advanced)
# This requires implementing PPO/DPO for chain-of-thought improvement
rl_config = {
    "base_model": "./qwen-physical-ai-sft",
    "reward_model": "spatial_temporal_reward_model", 
    "algorithm": "ppo",  # or "dpo"
    "num_episodes": 1000,
    "learning_rate": 5e-6,
    "ppo_epochs": 4,
    "batch_size": 64
}
```

## Step 4: Evaluate Results

### Test Chain-of-Thought Physical Reasoning
```python
# test_physical_reasoning.py
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Load finetuned model
model = Qwen2VLForConditionalGeneration.from_pretrained("./qwen-physical-ai-sft")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

def test_physical_reasoning(video_path):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": "Analyze the spatial relationships and temporal dynamics in this video. Think step by step about the physics involved."}
        ]
    }]
    
    inputs = processor.apply_chat_template(messages, return_tensors="pt")
    output = model.generate(
        **inputs, 
        max_new_tokens=512,  # Longer for reasoning chains
        do_sample=True, 
        temperature=0.1,
        top_p=0.9
    )
    return processor.decode(output[0], skip_special_tokens=True)

def test_spatial_understanding(video_path):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": "Describe the spatial layout and object relationships in this scene."}
        ]
    }]
    
    inputs = processor.apply_chat_template(messages, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(output[0], skip_special_tokens=True)
```

## Expected Improvements (Based on Cosmos-Reason1 Results)

### Before Physical AI Training
- Basic video descriptions
- Limited reasoning chains
- Surface-level spatial understanding
- Generic temporal descriptions

### After Physical AI Training  
- **Step-by-step reasoning**: "First, I observe... Then, considering physics... Therefore..."
- **Detailed spatial analysis**: Object positioning, depth relationships, scale consistency
- **Temporal dynamics**: Motion patterns, acceleration analysis, physics compliance
- **Causal reasoning**: Understanding cause-and-effect in physical interactions
- **Physics grounding**: References to real-world physics principles

## Success Metrics

1. **Reasoning Quality**: Length and accuracy of chain-of-thought explanations
2. **Physical Understanding**: Correct physics principles application
3. **Spatial Precision**: Accurate spatial relationship descriptions
4. **Temporal Accuracy**: Correct motion and timing analysis
5. **Consistency**: Stable reasoning across different video types

## Key Insights: Cosmos-Reason1 vs ALFA-VL

### Cosmos-Reason1 Insights:
- **Multi-stage training is crucial**: Each stage builds specific capabilities
- **Chain-of-thought data**: Essential for reasoning improvement  
- **Physical AI SFT**: Specialized training on spatial-temporal tasks shows significant gains
- **Longer sequences**: Reasoning chains require more tokens (2048+ vs 1024)
- **Lower learning rates**: Reasoning tasks benefit from more careful training (1e-5 vs 2e-5)
- **Reinforcement Learning**: Further improves reasoning chain quality

### ALFA-VL Adaptations:
- **Leverage pre-trained models**: Start with Qwen's strong foundation vs training from scratch
- **Real-world grounding**: Use Cosmos-Curate's real video data vs synthetic physics scenarios
- **Metadata-enhanced reasoning**: Incorporate motion/aesthetic scores into reasoning chains
- **Structured analysis format**: Spatial → Temporal → Physics progression
- **Efficient data curation**: Targeted video processing vs large-scale general curation
- **Domain-specific focus**: Optimize for spatial-temporal video understanding

## Next Steps

1. **Generate Cosmos-Curate data** with motion/aesthetic filtering
2. **Create Physical AI SFT dataset** with step-by-step reasoning chains
3. **Run multi-stage training** following Cosmos-Reason1 approach
4. **Evaluate reasoning capabilities** on spatial-temporal tasks
5. **Optional**: Implement RL stage for advanced reasoning improvement

## ALFA-VL Implementation Priority

**High Priority (Essential)**:
- ✅ **Physical AI SFT** with Cosmos-Curate chain-of-thought training data
- ✅ **Metadata integration** - motion/aesthetic scores in reasoning
- ✅ **Longer sequence lengths** (2048 tokens) for reasoning chains
- ✅ **Lower learning rates** (1e-5) for reasoning tasks
- ✅ **Structured reasoning format** (Spatial → Temporal → Physics)

**Medium Priority (Recommended)**:
- 🔄 **Multi-stage training** approach (basic SFT → Physical AI SFT)
- 🔄 **Comprehensive evaluation** on spatial-temporal reasoning tasks
- 🔄 **Real-world video benchmarks** vs synthetic physics tests

**Low Priority (Advanced)**:
- 🔄 **Reinforcement learning stage** (DPO/PPO for reasoning quality)
- 🔄 **Custom reward models** for spatial-temporal accuracy
- 🔄 **Multi-modal embeddings** integration (InternVideo2 + T5)

### ALFA-VL Success Criteria:
1. **Better than baseline Qwen 2.5 VL** on spatial-temporal tasks
2. **Structured reasoning chains** with clear Spatial → Temporal → Physics analysis
3. **Real-world applicability** on diverse video content
4. **Efficient training** with <10% of Cosmos-Reason1's data requirements