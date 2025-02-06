# Import necessary libraries
from huggingface_hub import login
import jsonlines
import random
from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTConfig, SFTTrainer

# Login to Hugging Face
huggingface_token = "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"
login(token=huggingface_token, add_to_git_credential=True)

# Define the path for the dataset
jsonl_path = '/mnt/ssd/1/sanguk/dataset/ko_commercial_train.jsonl'

# Load the dataset using JSONLines
indataset = []
with jsonlines.open(jsonl_path) as f:
    for lineno, line in enumerate(f.iter(), start=1):
        try:
            # Create a formatted template for instruction and response
            template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
            indataset.append(template.format(**line))
        except Exception as e:
            # If there's an issue, print the error and the line number
            print(f"Error at line {lineno}: {e}")

# Print message indicating that the dataset is successfully loaded
print('Dataset creation completed')

# Randomly sample 50,000 examples from the dataset for training
random.seed(42)  # Set seed for reproducibility
sampled_indataset = random.sample(indataset, 10000)

# Convert the dataset into Hugging Face's Dataset format
indataset = Dataset.from_dict({'text': sampled_indataset})

# Print the dataset info to confirm
print(indataset)

# Hugging Face model configuration
model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check for CUDA availability and set configurations accordingly
if torch.cuda.get_device_capability()[0] >= 8:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["lm_head"],
        llm_int8_enable_fp32_cpu_offload=True
    )
else:
    quantization_config = BitsAndBytesConfig()

# Configure SFT (Supervised Fine-Tuning)
sft_config = SFTConfig(
    output_dir="model",
    dataset_text_field="text",  # Set the text field
    max_seq_length=1024,        # Set the max sequence length
    packing=False               # Configure sequence packing based on input
)

# Configure TrainingArguments
training_params = TrainingArguments(
    output_dir=sft_config.output_dir,
    num_train_epochs=25,                            # Set the number of training epochs
    per_device_train_batch_size=8,                  # Set batch size per device
    gradient_accumulation_steps=1,                  # Set gradient accumulation steps
    optim="paged_adamw_8bit",                       # Optimizer
    save_steps=100000,                              # Save model every 100000 steps
    logging_steps=100000,                           # Log every 100000 steps
    learning_rate=2e-4,                             # Set learning rate
    weight_decay=0.001,                             # Set weight decay for regularization
    fp16=False,                                     # Disable float16 precision
    bf16=True,                                      # Enable bfloat16 precision
    max_grad_norm=0.3,                              # Max gradient norm for clipping
    max_steps=-1,                                   # No limit on number of steps
    warmup_ratio=0.03,                              # Warmup ratio for learning rate
    group_by_length=sft_config.packing,             # Group by length based on SFTConfig
    lr_scheduler_type="constant",                   # Use constant learning rate scheduler
    report_to="tensorboard"                         # Report to Tensorboard
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=indataset, 
    tokenizer=tokenizer,
    args=training_params,
    dataset_text_field=sft_config.dataset_text_field,  # Pass the dataset text field
    max_seq_length=sft_config.max_seq_length           # Pass the max sequence length
)

# Start training
trainer.train()
