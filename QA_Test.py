from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_trained_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(model, tokenizer, sequence, max_length):
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

model_path = "/Users/nihalcoskun/Desktop/CSE/Staj 2/Bittensor_relative_QA"
trained_model = load_trained_model(model_path)
tokenizer = load_tokenizer(model_path)

sequence =  "In the popular TV show \"Friends,\" what is the name of Joey's stuffed penguin?"
max_len = 100
generated_text = generate_text(trained_model, tokenizer, sequence, max_len)
print(generated_text)