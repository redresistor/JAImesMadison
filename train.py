import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        # Add special tokens for conversation format
        tokenizer.pad_token = tokenizer.eos_token
        
        # Read the pre-processed training data
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into individual QA pairs
        qa_pairs = text.split('\n\n')
        
        # Filter and clean QA pairs
        processed_texts = []
        for pair in qa_pairs:
            if 'Question:' in pair and 'Answer:' in pair:
                # Clean up any extra whitespace or formatting
                pair = re.sub(r'\s+', ' ', pair.strip())
                processed_texts.append(pair)
        
        # Encode all texts
        self.encodings = tokenizer(
            processed_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        self.attention_masks = self.encodings['attention_mask']
        # Store tensors directly
        self.input_ids = self.encodings['input_ids']

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx].clone().detach(),
            'attention_mask': self.attention_masks[idx].clone().detach(),
            'labels': self.input_ids[idx].clone().detach()
        }

    def __len__(self):
        return len(self.input_ids)

def train_model(model_name='gpt2', num_epochs=5, batch_size=4, learning_rate=5e-5):
    # Initialize model and tokenizer
    print("\n🔧 Initializing model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Prepare dataset and dataloader
    print("📚 Loading and preparing dataset...")
    train_dataset = TextDataset('trainingdata.txt', tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    model.to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_dataloader))

    # Training loop
    model.train()
    best_loss = float('inf')
    
    print("🚀 Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'📊 Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"💾 Saving best model with loss: {best_loss:.4f}")
            model.save_pretrained('madison_model')
            tokenizer.save_pretrained('madison_tokenizer')

    print("\n✨ Training completed!")
    print(f"🏆 Best loss achieved: {best_loss:.4f}")
    return model, tokenizer

if __name__ == "__main__":
    train_model()