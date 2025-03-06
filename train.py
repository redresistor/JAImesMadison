import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_cosine_schedule_with_warmup
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
        print(f"📊 Found {len(qa_pairs)} potential QA pairs")
        
        # Define patterns for questions and answers
        question_patterns = [
            r'(?i)question\s*:', 
            r'(?i)q\s*:', 
            r'(?i)query\s*:',
            r'(?i)inquiry\s*:'
        ]
        answer_patterns = [
            r'(?i)answer\s*:', 
            r'(?i)a\s*:', 
            r'(?i)response\s*:',
            r'(?i)reply\s*:'
        ]
        
        # Combine patterns into regex
        question_regex = '|'.join(question_patterns)
        answer_regex = '|'.join(answer_patterns)
        
        # Filter and clean QA pairs
        processed_texts = []
        skipped = 0
        
        for pair in qa_pairs:
            # Check if pair contains both question and answer
            has_question = bool(re.search(question_regex, pair))
            has_answer = bool(re.search(answer_regex, pair))
            
            if has_question and has_answer:
                # Clean up any extra whitespace or formatting
                pair = re.sub(r'\s+', ' ', pair.strip())
                
                # Standardize format to "Question:" and "Answer:"
                for pattern in question_patterns:
                    pair = re.sub(pattern, "Question:", pair, flags=re.IGNORECASE)
                for pattern in answer_patterns:
                    pair = re.sub(pattern, "Answer:", pair, flags=re.IGNORECASE)
                
                processed_texts.append(pair)
            else:
                skipped += 1
        
        print(f"✅ Processed {len(processed_texts)} valid QA pairs")
        print(f"⚠️ Skipped {skipped} invalid pairs")
        
        if processed_texts:
            print(f"📝 Sample QA pair: {processed_texts[0][:100]}...")
        
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

def train_model(model_name='gpt2-medium', num_epochs=5, batch_size=4, learning_rate=2e-5):
    # Initialize model and tokenizer
    print("\n🔧 Initializing model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # More robust CUDA detection
    if torch.cuda.is_available():
        print(f"🔍 CUDA detected: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("⚠️ CUDA not detected, using CPU")
        device = torch.device('cpu')

    if device.type == 'cuda':
        print(f"🧠 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"🔄 Current allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Prepare dataset and dataloader
    print("📚 Loading and preparing dataset...")
    full_dataset = TextDataset('trainingdata.txt', tokenizer)
    
    # Split into train and validation sets
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model.to(device)

    # Define optimizer and scheduler with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    try:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except NameError:
        print("⚠️ Cosine scheduler not available, using linear scheduler")
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    # Training loop
    model.train()
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    print("🚀 Starting training...")
    for epoch in range(num_epochs):
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        model.train()
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f'📊 Epoch {epoch+1}/{num_epochs}')
        print(f'   Training Loss: {avg_train_loss:.4f}')
        print(f'   Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"💾 Saving best model with validation loss: {best_val_loss:.4f}")
            model.save_pretrained('madison_model')
            tokenizer.save_pretrained('madison_tokenizer')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 Early stopping triggered!")
                break

    print("\n✨ Training completed!")
    print(f"🏆 Best validation loss achieved: {best_val_loss:.4f}")
    return model, tokenizer

if __name__ == "__main__":
    train_model()