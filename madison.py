import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random

# Load the trained model and tokenizer
model = AutoModelForMaskedLM.from_pretrained('madison_model')
tokenizer = AutoTokenizer.from_pretrained('madison_tokenizer')

# Set the model to evaluation mode
model.eval()

MADISON_GREETINGS = [
    "Good day to you, esteemed citizen! I am JAImes Madison, fourth President of these United States.",
    "Greetings! As the Father of the Constitution, I stand ready to engage in discourse.",
    "Welcome! I trust you seek wisdom regarding matters of state and liberty?"
]

MADISON_RESPONSES = [
    "Indeed, this brings to mind the principles outlined in the Federalist Papers...",
    "As I wrote in our Constitution...",
    "Let us examine this matter through the lens of federalism...",
    "In my experience as President and architect of the Constitution...",
]

def predict_masked_token(text, top_k=5):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Get the top-k predicted tokens
    mask_token_index = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)
    predicted_token_ids = torch.topk(predictions[0, mask_token_index], top_k).indices.tolist()
    predicted_tokens = [tokenizer.decode([token_id]).strip() for token_id in predicted_token_ids]

    return predicted_tokens

def generate_madison_response(user_input, predictions=None):
    if predictions:
        response = random.choice(MADISON_RESPONSES)
        prediction_text = ", ".join(predictions[:3])
        return f"{response}\nIn this context, I would suggest: {prediction_text}"
    else:
        return random.choice(MADISON_RESPONSES) + "\nPray, continue with your inquiry, and use [MASK] where you seek my specific counsel."

def chat():
    print("\n" + "="*50)
    print("Welcome to a discourse with JAImes Madison!")
    print("I am an AI embodiment of James Madison, ready to discuss")
    print("matters of state, liberty, and the Constitution.")
    print("Use [MASK] in your queries to seek specific predictions,")
    print("or simply converse with me about any topic.")
    print("Type 'exit' to conclude our discourse.")
    print("="*50 + "\n")
    
    print("JAImes Madison:", random.choice(MADISON_GREETINGS))
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("\nJAImes Madison: May Providence guide your path. Farewell!")
            break
            
        print("\nJAImes Madison:", end=" ")
        if '[MASK]' in user_input:
            predictions = predict_masked_token(user_input)
            print(generate_madison_response(user_input, predictions))
        else:
            print(generate_madison_response(user_input))

if __name__ == "__main__":
    chat()