import openai
import json
import time
from tqdm import tqdm
import re
from openai import OpenAI
from datetime import datetime

# Configure OpenAI client to use local endpoint
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

def chunk_text(text, max_tokens=3000):
    """Split text into chunks that won't exceed token limits."""
    # Rough estimate: 1 token ≈ 4 characters
    chars_per_chunk = max_tokens * 4
    
    # Split into smaller chunks
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_length + word_len > chars_per_chunk:
            if current_chunk:  # Only append if we have content
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len
    
    if current_chunk:  # Add the last chunk if it exists
        chunks.append(' '.join(current_chunk))
    
    return chunks

def print_qa_pair(qa_text, chunk_num, total_chunks):
    """Pretty print the Q&A pairs with a timestamp."""
    print("\n" + "="*80)
    print(f"Chunk {chunk_num}/{total_chunks} - {datetime.now().strftime('%H:%M:%S')}")
    print("-"*80)
    # Split into individual Q&A pairs and format them
    pairs = qa_text.split('\n\n')
    for pair in pairs:
        print(pair.strip())
        print("-"*40)
    print("="*80 + "\n")

def process_chunk(chunk):
    """Process a chunk of text using the local AI to create training examples."""
    try:
        # Create a shorter prompt for smaller chunks
        prompt = (
            "Convert this excerpt from the Federalist Papers into 2-3 question-answer pairs. "
            "Format: 'Question: [question]\nAnswer: As JAImes Madison, [answer]'\n\n"
        )
        
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are an expert at creating training data from historical texts."},
                {"role": "user", "content": f"{prompt}{chunk}"}
            ],
            temperature=0.7,
            max_tokens=800,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\nError processing chunk: {str(e)}")
        if "context length" in str(e).lower():
            print(f"Chunk length (chars): {len(chunk)}")
        return None

def main():
    print("\n🔍 Reading Federalist Papers...")
    try:
        with open('federalistpapers.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open('federalistpapers.txt', 'r', encoding='latin-1') as f:
            text = f.read()
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '"').replace('"', '"')
    text = re.sub(r'[^\w\s.,!?;:\']', ' ', text)
    
    print("📄 Splitting into chunks...")
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    print(f"📊 Created {total_chunks} chunks")
    
    print("\n🤖 Processing chunks with local AI...")
    training_data = []
    start_time = time.time()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📝 Processing chunk {i}/{total_chunks}...")
        result = process_chunk(chunk)
        if result:
            training_data.append(result)
            print_qa_pair(result, i, total_chunks)
            
            # Calculate and display progress statistics
            elapsed = time.time() - start_time
            avg_time_per_chunk = elapsed / i
            remaining_chunks = total_chunks - i
            estimated_remaining = avg_time_per_chunk * remaining_chunks
            
            print(f"⏱️  Progress: {i}/{total_chunks} chunks")
            print(f"⌛ Estimated time remaining: {estimated_remaining/60:.1f} minutes")
        
        time.sleep(0.5)
        
        # Save progress periodically
        if i % 5 == 0:  # Increased frequency of saves
            with open('trainingdata_partial.txt', 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(training_data))
            print(f"💾 Progress saved! ({i}/{total_chunks} chunks)")
    
    print("\n✨ Saving final processed training data...")
    with open('trainingdata2.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(training_data))
    
    total_time = time.time() - start_time
    print(f"\n✅ Done! Created trainingdata2.txt with {len(training_data)} examples")
    print(f"⏰ Total processing time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main() 