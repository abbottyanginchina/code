from openai import OpenAI
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/cfgs.yaml")

client = OpenAI(api_key=cfg.llm.api_key, 
                base_url=cfg.llm.base_url)

def chat_LLM(user_prompt):
    sys_prompt = ""
    response = client.chat.completions.create(
        model=cfg.llm.model,
        messages=[
            {'role': 'user', 
            'content': user_prompt},
            # {'role': 'system', 
            # 'content': sys_prompt}
        ]
    )

    return response.choices[0].message.content