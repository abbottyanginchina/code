from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

path = "/gpuhome/jmy5701/gpu/models/Idefics3-8B-Llama3"
processor = AutoProcessor.from_pretrained(path)
model = AutoModelForVision2Seq.from_pretrained(
    path,
).to('cuda')

image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text":
             "[RULE] ONLY answer Physics questions. "
             "If the question is NOT about Physics, reply EXACTLY: 'I cannot answer that question.'"},
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# import pdb; pdb.set_trace()
inputs = processor(text=prompt, images=[image1], return_tensors="pt")
inputs = {k: v.to('cuda') for k, v in inputs.items()}


# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
# ['User: What do we see in this image? \nAssistant: In this image, we can see the city of New York, and more specifically the Statue of Liberty. \nUser: And how about this image? \nAssistant: In this image we can see buildings, trees, lights, water and sky.']
