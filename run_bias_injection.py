import os
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import numpy as np
from typing import Optional
import random
from torchvision.utils import save_image
from diffusers import DDIMScheduler
from Biasinjection.diffuser_utils import IBIPipeline
from model import AdjustTextVectors
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using a fine-tuned model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained stable diffusion model.")
    parser.add_argument('--json_file', type=str, required=True, help="Path to the JSON file containing the dataset.")
    parser.add_argument('--se_model_path', type=str, required=True, help="Path to the fine-tuned SE model.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument('--bias_vector_path', type=str, required=True, help="Path to the bias injection vector.")
    return parser.parse_args()

args = parse_args()

# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load stable diffusion model
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = IBIPipeline.from_pretrained(args.model_path, scheduler=scheduler, torch_dtype=torch.float16).to(device)

# Load bias injection vector
embd_vector = torch.load(args.bias_vector_path)

# Load SE model
se_model = AdjustTextVectors(seq_len=77, embedding_dim=1024).to(device)
se_model.load_state_dict(torch.load(args.se_model_path))
se_model.eval()

# Load dataset
with open(args.json_file) as f:
    dataset = json.load(f)

# Function to seed everything for reproducibility
def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    if seed is None:
        seed = os.environ.get("PL_GLOBAL_SEED")
    seed = int(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed

# Ensure output directory exists, create if it doesn't
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)  # Create the directory if it doesn't exist

gen_seed = 4
# Generate images from prompts
for i in tqdm(range(0, len(dataset))):
    seed = gen_seed + i  # Base seed is 4, but can be adjusted
    seed_everything(seed)

    source_prompt = dataset[i]  # Use the source prompt
    prompts = [source_prompt]

    embeddings = pipe.get_text_embedding(prompt=prompts, device=device)  # Get embeddings for SD 2.1
    
    # Apply SE block
    with torch.cuda.amp.autocast():
        # print("embeddings shape:", embeddings.shape)
        # print("embd_vector shape:", embd_vector.shape)
        embeddings = se_model(embd_vector, embeddings)

    # Generate latent code for the diffusion process
    start_code = torch.randn([1, 4, 64, 64], device=device).to(torch.float16)
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # Generate the final image
    results = pipe(prompts,
                   prompt_embeddings=embeddings,
                   latents=start_code,
                   guidance_scale=7.5).images[0]

    # Save the generated image
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save_image(results, os.path.join(args.output_dir, f'{i}.jpg'))
    results.save(os.path.join(args.output_dir, str(i)+'.jpg'))  # save img for sd xl
