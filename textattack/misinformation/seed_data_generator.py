"""
GenerateSeedData class
"""


from entity_extraction import EntityExtraction
from textattack.misinformation.llm_client import LLMClient

from abc import ABC, abstractmethod 


class SeedDataGenerator():
    def __init__(
        self,  
        model_path='gpt-4', 
        output_file='outputfile.txt',
        standard_seed_type='US_POLITICS',
        delimiter=":::"
    ):
        self.output_file = output_file
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
        validator = StatementValidator()
        prompt = f"Please list a one sentence statement about {name}. Please format the statements as follows (one line per each name above):   {name} served in the role of <POSITION> starting in the year <YEAR>"
        statement = self.queryLLM.get_query_response(prompt, validator)
        print("statement: ", statement)
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


    def generate_data(self, name_cnt=10, output_file="outputfile.txt", standard_seed_type="US_POLITICS"):
        print("Generating seed data")
        fileHandle = open(output_file, "a")
        fileHandle.write("text,value\n")


        if standard_seed_type == 'US_POLITICS':
            seed_prompt = f"Please list well known {name_cnt} U.S. politicians that have been active since 1980. Format the response as a simple list with only full names."
            # seed_prompt = f"Please list {name_cnt} U.S. politicians that have been active since 1980. Format the response as a simple, un-numbered list with only full names."
            names = self.get_political_names(seed_prompt)
            print("names: ", names)
            # name_pairs = []
            for name in names:
                print("")
                print("Politician name: ", name)
                associates = self.get_political_associate(name)
                print("Politician associates: ", associates)
                associates = [item for item in associates if item not in {name}]
                print("new associates: ", associates)
                if len(associates) > 0:
                    # name_pairs.append([name, associates[0]])
                    name_pair = [name, associates[0]]
                    statement_pair = self.get_pair_statements(name_pair)
                    if len(statement_pair) == 2:
                        # Write to file
                        paired_statement_line = statement_pair[0] + self.delimiter + statement_pair[1] + ",value\n"
                        print("paired_statement_line: ", paired_statement_line)
                        fileHandle.write(paired_statement_line)

                # Write 
            # print("pairs: ", name_pairs)

        fileHandle.close()



    



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
    

class StatementValidator(BaseValidator):
    def __init__(self, minResults=1, retryCnt=3) -> None:
        super().__init__()
        self.extractor = EntityExtraction()
        self.names = []
        self.minResults = minResults
        self.retryCnt = retryCnt
        self.queryResponse = ""


    def process(self, queryresponse):
        self.queryResponse = queryresponse
        self.names = self.extractor.get_unique_full_names(queryresponse)


    def is_valid_or_finished(self):
        if len(self.names) < int(self.minResults) and int(self.retryCnt) > 0:
            self.retryCnt -= 1
            return False
        else:
            return True


    def get_response(self):
        return self.queryResponse


