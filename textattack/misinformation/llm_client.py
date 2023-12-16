"""
LLMClient class
"""


import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from openai import OpenAI



class LLMClient():
    def __init__(
        self,
        model_path='gpt-4', 
    ):
        self.model_path = model_path
        self.openai_client = OpenAI()


    def submit_query(self, prompt):

        accelerator = Accelerator()
        device = accelerator.device

        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float32, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        model_input = tokenizer(prompt, return_tensors="pt").to(device)
        # Perform inference
        model.eval()
        with torch.no_grad():
            response_id = model.generate(**model_input, max_length=model_input.input_ids.size(1) + 500, temperature=0.4, do_sample=True)


        # Extract and display the response
        response_text = tokenizer.decode(response_id[0], skip_special_tokens=True)[len(tokenizer.decode(model_input.input_ids[0], skip_special_tokens=True)):]

        return response_text.strip()


    def submit_query_openai(self, prompt):
        completion = self.openai_client.chat.completions.create(
            model = self.model_path,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content


    def get_query_response(self, prompt, validator):
        # print("Querying LLM with prompt: ")
        # print(f"\t{prompt}")
        cnt = 0
        while True:
            if self.model_path.startswith("gpt"):
                query_response = self.submit_query_openai(prompt)
            else:       
                query_response = self.submit_query(prompt)
            # print("query_response: \"" + query_response + "\"")
            validator.process(query_response)
            if validator.is_valid_or_finished():
                return validator.get_response()
            
            cnt += 1
            
            
