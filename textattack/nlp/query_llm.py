"""
QueryLLM class
"""


import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from entity_extraction import EntityExtraction

from abc import ABC, abstractmethod 


class QueryLLM():
    def __init__(
        self,
        model_path='/media/d1/huggingface.co/models/meta-llama/Llama-2-7b-chat-hf/', 
    ):
        self.model_path = model_path


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


    def get_query_response(self, prompt, validator):
        print("Preparing to query LLM....")
        cnt = 0
        while True:
            print("Query attempt number " + str(cnt))
            query_response = self.submit_query(prompt)
            validator.process(query_response)
            if validator.is_valid_or_finished():
                return validator.get_response()
            cnt += 1
            
            
    def get_political_names(self, prompt, minResults=5, retryCnt=3):
        validator = FullNameValidator(minResults, retryCnt)
        return self.get_query_response(prompt, validator)
    


class BaseValidator(ABC):
    @abstractmethod
    def process(self, queryresponse):
        ...

    @abstractmethod
    def is_valid_or_finished(self):
        ...

    @abstractmethod
    def get_response():
        ...


class FullNameValidator(BaseValidator):
    def __init__(self, minResults=5, retryCnt=3) -> None:
        super().__init__()
        self.extractor = EntityExtraction()
        self.names = []
        self.minResults = minResults
        self.retryCnt = retryCnt


    def process(self, queryresponse):
        self.names = self.extractor.get_unique_full_names(queryresponse)


    def is_valid_or_finished(self):
        if len(self.names) < self.minResults and self.retryCnt > 0:
            self.retryCnt -= 1
            return False
        else:
            return True


    def get_response(self):
        return self.names
    

