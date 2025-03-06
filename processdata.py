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

def test_api_connection():
    """Test the connection to the local API."""
    try:
        print("🔍 Testing API connection...")
        response = client.chat.completions.create(
            model="local-model",  # We'll try to get available models if this fails
            messages=[
                {"role": "user", "content": "Hello, are you working?"}
            ],
            max_tokens=10
        )
        print("✅ API connection successful!")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {str(e)}")
        try:
            # Try to list available models
            print("\n🔍 Attempting to list available models...")
            models = client.models.list()
            print("📋 Available models:")
            for model in models:
                print(f"   - {model.id}")
            return False
        except Exception as e2:
            print(f"❌ Could not list models: {str(e2)}")
            return False

def chunk_text(text, max_tokens=2000):
    """Split text into meaningful chunks by paragraphs and arguments."""
    # More flexible pattern matching
    patterns = [
        r'(?i)(federalist\.?\s*(?:no\.?|number\.?)?\s*\d+)',  # Matches various "Federalist No." formats
        r'(?i)(the\s+federalist\.?\s*(?:no\.?|number\.?)?\s*\d+)',
        r'(?i)(federalist\s+papers?\s*(?:no\.?|number\.?)?\s*\d+)'
    ]
    
    # Try each pattern and use the one that finds matches
    for pattern in patterns:
        print(f"\n🔍 Trying pattern: {pattern}")
        papers = re.split(pattern, text)
        if len(papers) > 1:
            print(f"✅ Found {(len(papers)-1)//2} papers with this pattern!")
            break
    else:
        print("❌ No papers found with any pattern!")
        print("\n📄 First 200 characters of text:")
        print(text[:200])
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i in range(1, len(papers), 2):
        if i + 1 >= len(papers):
            break
            
        title = papers[i].strip()
        content = papers[i + 1].strip() if i + 1 < len(papers) else ""
        
        # Debug output
        print(f"\n📄 Processing {title}")
        print(f"   Content length: {len(content)} characters")
        
        # Estimate token length (rough approximation)
        combined_length = len(title + content) // 4
        
        if combined_length > max_tokens:
            # Split large papers into argument-based chunks
            paragraphs = content.split('\n\n')
            current_para_chunk = []
            current_para_length = len(title) // 4
            
            for para in paragraphs:
                para_length = len(para) // 4
                if current_para_length + para_length > max_tokens:
                    chunk_text = title + '\n\n' + '\n\n'.join(current_para_chunk)
                    chunks.append(chunk_text)
                    print(f"   Created chunk of length: {len(chunk_text)} characters")
                    current_para_chunk = [para]
                    current_para_length = len(title) // 4 + para_length
                else:
                    current_para_chunk.append(para)
                    current_para_length += para_length
            
            if current_para_chunk:
                chunk_text = title + '\n\n' + '\n\n'.join(current_para_chunk)
                chunks.append(chunk_text)
                print(f"   Created final chunk of length: {len(chunk_text)} characters")
        else:
            chunk_text = title + '\n\n' + content
            chunks.append(chunk_text)
            print(f"   Created single chunk of length: {len(chunk_text)} characters")
    
    return chunks

def process_chunk(chunk, model_name="local-model"):
    """Process a chunk of text using the local AI to create training examples."""
    try:
        # Create a more focused prompt for Madison's style
        prompt = (
            "Convert this excerpt from the Federalist Papers into a series of Q&A pairs that capture "
            "Madison's logical reasoning and argumentation style. Each pair should follow this format:\n"
            "Question: [specific question about federalism, democracy, or constitutional principles]\n"
            "Answer: [reasoned response with clear arguments "
            "and specific examples, maintaining Madison's formal yet persuasive style]\n\n"
            "Focus on:\n"
            "1. Clear logical progression of ideas\n"
            "2. Specific examples and historical references\n"
            "3. Careful consideration of counterarguments\n"
            "4. Application to democratic principles\n\n"
        )
        
        print(f"\n🤖 Sending request to API...")
        print(f"   Model: {model_name}")
        print(f"   Chunk size: {len(chunk)} characters")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": (
                    "You are JAImes Madison, primary architect of the Constitution and author of many "
                    "Federalist Papers. You excel at careful reasoning, detailed argumentation, and "
                    "the application of historical examples to constitutional principles."
                )},
                {"role": "user", "content": f"{prompt}Text to convert:\n\n{chunk}"}
            ],
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9
        )
        
        result = response.choices[0].message.content
        print(f"✅ Successfully processed chunk, generated {len(result)} characters")
        return result
    except Exception as e:
        print(f"\n❌ Error processing chunk: {str(e)}")
        if "context length" in str(e).lower():
            print(f"   Chunk length (chars): {len(chunk)}")
        return None

def main():
    # First test the API connection
    if not test_api_connection():
        print("\n❌ Please check your local API setup and try again.")
        return

    print("\n📚 Reading Federalist Papers...")
    try:
        with open('federalistpapers.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            print(f"✅ Successfully read file, {len(text)} characters")
    except UnicodeDecodeError:
        print("⚠️ UTF-8 encoding failed, trying latin-1...")
        with open('federalistpapers.txt', 'r', encoding='latin-1') as f:
            text = f.read()
            print(f"✅ Successfully read file with latin-1 encoding, {len(text)} characters")
    
    # Clean the text while preserving important structure
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '"').replace('"', '"')
    text = re.sub(r'[^\w\s.,!?;:\'\n\-]', ' ', text)
    
    print("\n📄 Splitting into chunks by argument structure...")
    chunks = chunk_text(text)
    print(f"📊 Created {len(chunks)} logically structured chunks")
    
    # Test process with first chunk
    print("\n🧪 Testing processing with first chunk...")
    test_result = process_chunk(chunks[0])
    if test_result is None:
        print("❌ Initial test failed. Please check the error messages above.")
        return
    print("✅ Initial test successful!")
    
    print("\n🚀 Processing all chunks...")
    training_data = []
    start_time = time.time()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📝 Processing chunk {i}/{len(chunks)}...")
        result = process_chunk(chunk)
        if result:
            training_data.append(result)
            
            # Calculate and display progress statistics
            elapsed = time.time() - start_time
            avg_time_per_chunk = elapsed / i
            remaining_chunks = len(chunks) - i
            estimated_remaining = avg_time_per_chunk * remaining_chunks
            
            print(f"⏱️  Progress: {i}/{len(chunks)} chunks")
            print(f"⌛ Estimated time remaining: {estimated_remaining/60:.1f} minutes")
            
            # Save progress after each successful chunk
            with open('trainingdata_partial2.txt', 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(training_data))
            print(f"💾 Progress saved! ({i}/{len(chunks)} chunks)")
            
            # Display sample of processed text
            print("\n🔍 Sample of processed text:")
            print("-" * 40)
            print(result[:200] + "...")
            print("-" * 40)
        else:
            print(f"⚠️ Failed to process chunk {i}, skipping...")
        
        time.sleep(0.5)
    
    if not training_data:
        print("\n❌ No data was successfully processed. Please check the errors above.")
        return
    
    print("\n✨ Saving final processed training data...")
    with open('trainingdata2.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(training_data))
    
    total_time = time.time() - start_time
    print(f"\n✅ Done! Created trainingdata2.txt with {len(training_data)} examples")
    print(f"⏰ Total processing time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main() 