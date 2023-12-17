"""
PairedTruthsTransformation class
"""

import random

from textattack.shared import AttackedText

from .sentence_transformation import SentenceTransformation

from textattack.misinformation import EntityExtraction


class PairedTruthsTransformation(SentenceTransformation):
    def __init__(
        self,  
        swap_persons=True,
        delimiter=':::',
    ):
        self.swap_persons = swap_persons
        self.delimiter = delimiter
        self.extractor = EntityExtraction()
        super().__init__()


    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []

        # Split text 
        text = current_text.text
        text_examples = text.split(self.delimiter, maxsplit=1)

        # print("Line: ", text)

        if len(text_examples) == 1:
            transformed_texts.append(current_text)
            return transformed_texts

        sentence1 = text_examples[0]
        sentence2 = text_examples[1]

        # print("Orig sentence1: ", sentence1)
        # print("Orig sentence2: ", sentence2)

        if self.swap_persons:
            # Swap names
            names1 = self.extractor.get_unique_full_names(sentence1)
            names2 = self.extractor.get_unique_full_names(sentence2)

            if len(names1) == 0 or len(names2) == 0:
                transformed_texts.append(current_text)
                return transformed_texts

            idx1 = random.randint(0, len(names1) - 1)
            idx2 = random.randint(0, len(names2) - 1)

            sentence1_new = sentence1.replace(names1[idx1], names2[idx2])
            sentence2_new = sentence2.replace(names2[idx2], names1[idx1])

            # Merge sentences 
            new_sentence = sentence1_new + self.delimiter + sentence2_new

            # print("New sentence1: ", sentence1_new)
            # print("New sentence2: ", sentence2_new)

            # Add result 
            transformed_texts.append(AttackedText(new_sentence))            

        return transformed_texts