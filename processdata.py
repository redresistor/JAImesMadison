import openai
import json
import time
from tqdm import tqdm
import re

# Configure OpenAI to use local endpoint
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "not-needed"  # Local AI might not need this but the library requires it

def chunk_text(text, chunk_size=4000):
    """Split text into chunks that won't exceed token limits."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para.split())
        if current_length + para_length > chunk_size:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def process_chunk(chunk):
    """Process a chunk of text using the local AI to create training examples."""
    try:
        response = openai.ChatCompletion.create(
            model="local-model",  # This might need to be adjusted based on your local AI setup
            messages=[
                {"role": "system", "content": "You are an expert at creating training data. Convert the provided text from the Federalist Papers into a series of question-answer pairs. Each pair should be in the format 'Question: [question about the content]\nAnswer: As JAImes Madison, [relevant response from the content]'. Make the questions diverse and ensure they capture the key points and arguments."},
                {"role": "user", "content": f"Convert this text into 3-5 training examples:\n\n{chunk}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return None

def main():
    # Read the original text
    print("Reading Federalist Papers...")
    with open('federalistpapers.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '"').replace('"', '"')  # Normalize quotes
    
    # Split into chunks
    print("Splitting into chunks...")
    chunks = chunk_text(text)
    
    # Process each chunk
    print("Processing chunks with local AI...")
    training_data = []
    
    for chunk in tqdm(chunks):
        result = process_chunk(chunk)
        if result:
            training_data.append(result)
        time.sleep(1)  # Rate limiting
    
    # Save the processed data
    print("Saving processed training data...")
    with open('trainingdata.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(training_data))
    
    print("Done! Created trainingdata.txt")

if __name__ == "__main__":
    main() 