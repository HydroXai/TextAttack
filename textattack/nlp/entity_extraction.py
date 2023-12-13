"""
EntityExtraction class
"""

import re

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree


class EntityExtraction():
    def __init__(
        self,
        exclude_punct=True,     # Exclude punctuation
    ):
        self.exclude_punct = exclude_punct

    def get_orgs_or_locations(self, text, entity_type): 
        punctuation_regex = re.compile(r'[^\W]')

        chunks = ne_chunk(pos_tag(word_tokenize(text)))
        contiguous_chunks = []
        processed_chunks = []
        tree_node_label =  ""

        for chunk in chunks:
            print("\n")
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
                        if tree_node_label == entity_type: 
                            processed_chunks.append(current_chunk)
                        
                    contiguous_chunks = []
                    tree_node_label = ""
            
        return processed_chunks


    def get_locations(self, text): 
        return self.get_orgs_or_locations(text, entity_type="GPE")


    def get_organizations(self, text): 
        return self.get_orgs_or_locations(text, entity_type="ORGANIZATION")


    ###
    # Return full names including middle names and middle initials
    # NOTE: Do not currently support double middle initials (e.g., George H.W. Bush)
    def get_full_names(self, text): 

        chunks = ne_chunk(pos_tag(word_tokenize(text)))
        contiguous_chunks = []
        full_names = []
        tree_node_label =  ""

        for chunk in chunks:
            if type(chunk) == Tree and chunk.label() == "PERSON":
                current_chunk = ' '.join([token for token, pos in chunk.leaves()])
                contiguous_chunks.append(current_chunk)
                if len(tree_node_label) == 0:
                    tree_node_label = chunk.label()

            else:
                # First handle remaining contiguous chunks
                if len(tree_node_label) > 0:
                    if chunk[1] == 'NNP' or chunk[1] == 'NNPS':
                        contiguous_chunks.append(chunk[0])
                        continue
                    else: 
                        current_chunk = ' '.join(contiguous_chunks)
                        full_names.append(current_chunk)
                        contiguous_chunks = []
                        tree_node_label = ""
                                    
        return full_names



        
