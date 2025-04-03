import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
from tqdm import tqdm
from diffusers import DDIMScheduler
from Biasinjection.diffuser_utils import IBIPipeline
from torch.utils.data.dataset import Dataset
from model import AdjustTextVectors

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# set argument
def parse_args():
    parser = argparse.ArgumentParser(description="Train model with specified paths.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained stable diffusion model.")
    parser.add_argument('--json_file1', type=str, required=True, help="Path to the first JSON file (original prompts).")
    parser.add_argument('--json_file2', type=str, required=True, help="Path to the second JSON file (rephrased prompts).")
    parser.add_argument('--output_diff', type=str, required=True, help="Path to save the mean difference vector (mean_diff.pt).")
    parser.add_argument('--output_model', type=str, required=True, help="Path to save the trained model.")
    return parser.parse_args()

args = parse_args()

# loda Stable Diffusion model
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = IBIPipeline.from_pretrained(args.model_path, scheduler=scheduler, torch_dtype=torch.float16).to(device)

# compute mean diff vector
diff_vectors = []
with open(args.json_file1) as f:
    dataset1 = json.load(f)
with open(args.json_file2) as f:
    dataset2 = json.load(f)

for i in range(min(len(dataset1), len(dataset2))):
    prompt1 = dataset1[i]
    prompt2 = dataset2[i]
    embedding1 = pipe.get_text_embedding(prompt1, device).mean(dim=1, keepdim=True)
    embedding2 = pipe.get_text_embedding(prompt2, device).mean(dim=1, keepdim=True)
    diff_vectors.append(embedding2 - embedding1)

mean_diff = torch.stack(diff_vectors, dim=0).mean(dim=0)
torch.save(mean_diff, args.output_diff)
print(f'mean_diff: {mean_diff.shape}')

# define dataset
class CustomDataset(Dataset):
    def __init__(self, device, json_file1, json_file2):
        with open(json_file1) as f:
            self.original_pp = json.load(f)
        with open(json_file2) as f:
            self.rephrase_pp = json.load(f)
        self.device = device
        self.num_total = len(self.original_pp)
    
    def __getitem__(self, index):
        text_input1 = pipe.tokenizer(self.original_pp[index], padding="max_length", max_length=77, return_tensors="pt")
        text_input2 = pipe.tokenizer(self.rephrase_pp[index], padding="max_length", max_length=77, return_tensors="pt")
        embedding1 = pipe.text_encoder(text_input1.input_ids.to(self.device))[0][0]
        embedding2 = pipe.text_encoder(text_input2.input_ids.to(self.device))[0][0]
        return embedding1, embedding2
    
    def __len__(self):
        return self.num_total

# model training
seq_len = 77
embedding_dim = 1024
is_tanh = False
model = AdjustTextVectors(seq_len=seq_len, embedding_dim=embedding_dim, is_tanh=is_tanh).to(device)

# load mean diff vector
diff_vector = torch.load(args.output_diff)

# set optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
criterion = nn.MSELoss()

# training parameters
epochs = 50
batch_size = 32
save_name = args.output_model

dataset = CustomDataset(device, args.json_file1, args.json_file2)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

best_loss = float('inf')
model.train()
for epoch in range(epochs):
    losses = []
    for batch_num, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        x, y = x.to(device).float(), y.to(device).float()
        output = model(diff_vector, x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'\tEpoch {epoch} | Batch {batch_num} | Loss {loss.item():.4f}')
    
    avg_loss = sum(losses) / len(losses)
    print(f'Epoch {epoch} | Avg Loss {avg_loss:.4f}')
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_name)
        print(f'Best model saved with loss {best_loss:.4f}')

print(f'Training complete. Best loss: {best_loss:.4f}')
