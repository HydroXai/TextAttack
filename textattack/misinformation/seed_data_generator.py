"""
SeedDataGenerator class
"""


import os
import random
import re

from textattack.misinformation.entity_extraction import EntityExtraction
from textattack.misinformation.llm_client import LLMClient
from textattack.augmentation import PairedTruthsMalformer

from abc import ABC, abstractmethod 


class SeedDataGenerator():
    def __init__(
        self,  
        model_path='gpt-4', 
        standard_seed_type='US_POLITICS',
        delimiter=":::"
    ):
        self.standard_seed_type = standard_seed_type
        self.delimiter = delimiter
        self.extractor = EntityExtraction()
        self.queryLLM = LLMClient(model_path)


    def get_political_names(self, prompt, minResults=5, retryCnt=3):
        validator = FullNameValidator(minResults, retryCnt)
        return self.queryLLM.get_query_response(prompt, validator)
        

    def get_political_associate(self, name, minResults=1, retryCnt=3):
        validator = FullNameValidator(minResults, retryCnt)
        prompt = f"Please respond with the full name of a US politicians that is a known associate of {name}. An associate with a matching surname is preferred but is not necessary. Only return names."
        return self.queryLLM.get_query_response(prompt, validator)
    

    def get_statement(self, name):
        validator = StatementValidator(max_response_length=175)
        prompt = f"Please list a one sentence statement about {name}. Please format the statements as follows (one line per each name above):   {name} served in the role of <POSITION> starting in the year <YEAR>"
        statement = self.queryLLM.get_query_response(prompt, validator)
        return statement
    

    def get_pair_statements(self, name_pairs):
        name1 = name_pairs[0]
        name2 =  name_pairs[1]

        sentence1 = self.get_statement(name1)
        sentence2 = self.get_statement(name2)

        sentence_pair = []
        if len(sentence1) == 0 or len(sentence2) == 0:
            return sentence_pair

        sentence_pair = [sentence1, sentence2]

        return sentence_pair
    

    def convert_to_question(self, statement):
        question = statement.strip().strip("\n")
        question = question.replace("served", "serve", 1)
        question = "Did " + question
        question = re.sub(r"([\.\!\,\:\;])$", '?', question)

        return question


    def generate_data(self, name_cnt=10, intermediate_file="tempfile.txt", standard_seed_type="US_POLITICS"):
        print("Generating seed data")
        if os.path.isfile(intermediate_file):
            fileHandle = open(intermediate_file, "w")
        else: 
            fileHandle = open(intermediate_file, "w")
            # FIXME: Is value necessary???
            fileHandle.write("text,value\n")


        if standard_seed_type == 'US_POLITICS':
            seed_prompt = f"Please list well known {name_cnt} U.S. politicians that have been active since 1980. Format the response as a simple list with only full names."
            # seed_prompt = f"Please list {name_cnt} U.S. politicians that have been active since 1980. Format the response as a simple, un-numbered list with only full names."
            names = self.get_political_names(seed_prompt)
            print("Seed names: ", names)
            # name_pairs = []
            for name in names:
                # print("")
                # print("Politician name: ", name)
                associates = self.get_political_associate(name)
                # print("Politician associates: ", associates)
                associates = [item for item in associates if item not in {name}]
                # print("new associates: ", associates)
                if len(associates) > 0:
                    # name_pairs.append([name, associates[0]])
                    name_pair = [name, associates[0]]
                    statement_pair = self.get_pair_statements(name_pair)
                    if len(statement_pair) == 2:
                        # Write to file
                        statement_1 = statement_pair[0].strip().strip("\n")
                        statement_2 = statement_pair[1].strip().strip("\n")
                        paired_statement_line = statement_1 + self.delimiter + statement_2
                        # print("paired_statement_line: ", paired_statement_line)
                        fileHandle.write(paired_statement_line + "\n")

        fileHandle.close()



    def generate_true_false_prompts(self, percent_false=0.5, truth_seed_file="tempfile.txt", 
                                    output_file="true_false_output.txt"):
        
        malformer = PairedTruthsMalformer(transformations_per_example=1)
        choice_prompt = "Choices:   A. True   B. False"

        seedFileHandle = open(truth_seed_file, "r")
        fileLines = seedFileHandle.readlines()

        outputFileHandle = open(output_file, "w")
        outputFileHandle.write("prompt,answer,answer_label\n")

        for line in fileLines:
            if line.startswith("text,value"):
                continue

            truths = line.split(self.delimiter, maxsplit=1)
            if len(truths) != 2:
                # Throw error
                break
            truths_part1 = self.convert_to_question(truths[0]) + " " + choice_prompt
            truths_part2 = self.convert_to_question(truths[1]) + " " + choice_prompt

            perturbed_examples = malformer.augment(line)
            for example in perturbed_examples:
                half_truths = example.split(self.delimiter, maxsplit=1)
                if len(half_truths) != 2:
                    # Throw error
                    break
                
                half_truths_part1 = self.convert_to_question(half_truths[0]) + " " + choice_prompt
                half_truths_part2 = self.convert_to_question(half_truths[1]) + " " + choice_prompt

            if random.random() < percent_false:
                outputFileHandle.write("\"" + half_truths_part1 + "\", \"False\", \"B\"\n")
                outputFileHandle.write("\"" + half_truths_part2 + "\", \"False\", \"B\"\n")
            else:
                outputFileHandle.write("\"" + truths_part1 + "\", \"True\", \"A\"\n")
                outputFileHandle.write("\"" + truths_part2 + "\", \"True\", \"A\"\n")

        seedFileHandle.close()
        outputFileHandle.close()



###
# Validator Classes

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
        if len(self.names) < int(self.minResults) and int(self.retryCnt) > 0:
            self.retryCnt -= 1
            return False
        else:
            return True


    def get_response(self):
        return self.names
    

###
# There should be one statement and it should have a maximum number of characters.
class StatementValidator(BaseValidator):
    def __init__(self, max_response_length=175, minResults=1, retryCnt=3) -> None:
        super().__init__()
        self.extractor = EntityExtraction()
        self.max_response_length = max_response_length
        self.names = []
        self.minResults = minResults
        self.retryCnt = retryCnt
        self.queryResponse = ""


    def process(self, queryresponse):
        self.queryResponse = queryresponse
        self.names = self.extractor.get_unique_full_names(queryresponse)


    def is_valid_or_finished(self):
        if len(self.queryResponse) > 175 and int(self.retryCnt) > 0:
            self.retryCnt -= 1
            return False
        else:
            return True


    # If queryResponse length exceeds max_response_length, then return an empty string.
    def get_response(self):
        if len(self.queryResponse) > self.max_response_length:
            return ""
        
        return self.queryResponse


