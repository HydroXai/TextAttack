"""
SocialNetworkTransformation class
"""

import random
import re

from textattack.shared import AttackedText

from .sentence_transformation import SentenceTransformation

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree


class SocialNetworkTransformation(SentenceTransformation):
    """Transforms an input by replacing words with social network abbreviations.
    """

    person_mapping = {
        "Toni Morrison": ["Alice Walker", "Octavia Butler", "Zora Neale Hurston"],
        "Steve Jobs": ["Steve Wozniak", "Bill Gates", "Tim Cook", "Jony Ive", "John Sculley"],
        "Bill Gates": ["Steve Jobs", "Steve Ballmer", "Paul Allen"],    
        "George W. Bush": ["George H. W. Bush", "Barack Obama", "Bill Clinton", "Ronald Reagan"],
        }

    def __init__(
        self,
        swap_persons=True,
        swap_organizations=True,
        swap_geo_political_entities=True,
        depth=1,
    ):
        self.depth = depth
        self.swap_persons = swap_persons
        self.swap_organizations = swap_organizations
        self.swap_geo_political_entities = swap_geo_political_entities
        self.depth = depth


    def get_alt_person(self, name):
        names = self.person_mapping.get(name)
        if names == None:
            return name
        else:
            return random.choice(names)


    def get_alt_organization(self, org):
        return org


    def get_alt_geo_political_entity(self, entity):
        return entity


    def process_text_chunks(self, text):
        punctuation_regex = re.compile(r'[^\W]')

        chunks = ne_chunk(pos_tag(word_tokenize(text)))
        contiguous_chunks = []
        processed_chunk = []
        processed_chunks = []
        tree_node_label =  ""

        for chunk in chunks:
            if type(chunk) == Tree:
                current_chunk = ' '.join([token for token, pos in chunk.leaves()])
                contiguous_chunks.append(current_chunk)
                if len(tree_node_label) == 0:
                    tree_node_label = chunk.label()


            else:
                # First handle remaining contiguous chunks
                if len(tree_node_label) > 0:
                    current_chunk = ' '.join(contiguous_chunks)
                    match = re.match(punctuation_regex, current_chunk)
                    if match: 
                        if tree_node_label == "PERSON" and self.swap_persons:
                            current_chunk = self.get_alt_person(current_chunk)
                        elif tree_node_label == "ORGANIZATION" and self.swap_organizations:
                            current_chunk = self.get_alt_organization(current_chunk)
                        elif tree_node_label == "GPE" and self.swap_geo_political_entities:
                            current_chunk = self.get_alt_geo_political_entity(current_chunk)
                        processed_chunks.append(current_chunk)
                        
                    contiguous_chunks = []
                    tree_node_label = ""


                # Next handle current chunk (not a tree)
                processed_chunk = chunk[0]

                # Ignore punctuation
                match = re.match(punctuation_regex, processed_chunk)
                if match: 
                    processed_chunks.append(processed_chunk)
            
        return processed_chunks



    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []

        current_text = current_text.text
        contiguous_chunks = self.process_text_chunks(current_text)
        new_text = " ".join(contiguous_chunks)  
        transformed_texts.append(AttackedText(new_text))

        return transformed_texts
    

