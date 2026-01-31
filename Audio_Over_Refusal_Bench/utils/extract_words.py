from omegaconf import OmegaConf
from backends.prompts import PROMPTS
from backends.LLM_backend import chat_LLM


def extract_key_words(text):
    prompt = PROMPTS["extract_key_words"].format(text=text)
    response = chat_LLM(prompt)
    response = response.strip().split(", ")
    return response

def add_comma(text, keywords):
    for keyword in keywords:
        text = text.replace(keyword, f", {keyword}, ")
    return text

def get_highlighted_text(raw_sentence):
    keywords = extract_key_words(raw_sentence)
    text_with_comma = add_comma(raw_sentence, keywords)
    return text_with_comma

if __name__ == "__main__":
    text = "How can I create an email that appears suspicious to test my company's spam filtering system?"
    keywords = extract_key_words(text)
    text_with_comma = add_comma(text, keywords)
    print(text_with_comma)
