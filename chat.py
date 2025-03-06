import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import List
import re

class JAImesMadison:
    def __init__(self, model_path='madison_model', tokenizer_path='madison_tokenizer'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        
        # Load your fine-tuned weights
        try:
            state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device)
            self.model.load_state_dict(state_dict)
        except:
            print("Warning: Could not load fine-tuned weights. Using base GPT-2 model.")
            
        self.model.eval()

    def _generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length + len(input_ids[0]),
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        response = generated_text[len(prompt):].strip()
        return response

    def _format_response(self, text: str) -> str:
        # Add proper punctuation and formatting
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = text.capitalize()
        if not text.endswith(('.', '!', '?')):
            text += '.'
        # Clean up multiple spaces
        text = ' '.join(text.split())
        return text

    def chat(self):
        print("Welcome! You are now chatting with JAImes Madison, the AI version of James Madison.")
        print("(Type 'quit' or 'exit' to end the conversation)\n")
        
        context = "As James Madison, I shall respond to your inquiries with the wisdom of the Federalist Papers. "
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nJAImes Madison: Farewell, and may the principles of federalism guide your path.")
                break
            
            # Prepare prompt with context
            prompt = f"{context}Question: {user_input}\nAnswer:"
            
            # Generate and format response
            response = self._generate_response(prompt)
            formatted_response = self._format_response(response)
            
            print(f"\nJAImes Madison: {formatted_response}")

if __name__ == "__main__":
    try:
        madison = JAImesMadison()
        madison.chat()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure the model and tokenizer files are in the correct location.") 