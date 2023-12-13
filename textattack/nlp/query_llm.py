"""
QueryLLM class
"""


import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from entity_extraction import EntityExtraction


class QueryLLM():
    def __init__(
        self,
        model_path='/media/d1/huggingface.co/models/meta-llama/Llama-2-7b-chat-hf/', 
    ):
        self.model_path = model_path


    def get_query_response(self, prompt):

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


    def get_political_names(self, prompt, minResults=5, retryCnt=3):
        extractor = EntityExtraction()

        names = []
        while True:
            response = self.get_query_response(prompt)
            names = extractor.get_unique_full_names(response)

            if len(names) < minResults and retryCnt > 0:
                retryCnt =- 1
                continue
            break

        return names

