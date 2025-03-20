from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def generate_explanation_gpt2(letter, sentiment):
    # Prepare the input prompt for GPT-2 with a clearer directive
    input_text = f"Given that the sentiment of the following letter is {sentiment}, explain why the sentiment is {sentiment} based on the content of the letter: {letter}"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

    # Generate the explanation using GPT-2
    output_ids = model.generate(input_ids, max_length=150, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)

    # Decode the generated explanation
    explanation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return explanation


# Example letter and sentiment
letter = """Dear Council, I am writing to bring attention to the recent increase in littering in our neighborhood. In the past few months, I've noticed an alarming amount of trash being left in public areas, especially near the park and along Main Street. The current trash bins are overflowing and appear to be insufficient for the amount of waste. This is not only unsightly but also poses a serious health risk to local residents. I urge the council to increase the number of bins and arrange for more frequent waste collection. Thank you for your attention to this matter."""
sentiment = "negative"

# Generate explanation for the sentiment
explanation = generate_explanation_gpt2(letter, sentiment)
print(explanation)
