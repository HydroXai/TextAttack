"""
EntityExtraction class
"""

import re

from nltk import ne_chunk, pos_tag, word_tokenize, tree2conlltags
from nltk.tree import Tree


class EntityExtraction():
    def __init__(
        self,
        exclude_punct=True,     # Exclude punctuation
    ):
        self.exclude_punct = exclude_punct

    def get_orgs_or_locations(self, text, entity_type): 
        punctuation_regex = re.compile(r'[^\W]')   # Not alphanumeric (will match with punctuation)

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


    def is_proper_noun(self, text):
        punctuation_regex = re.compile(r'[\W]')    # Non-alphanumeric
        chunks = ne_chunk(pos_tag(word_tokenize(text)))

        for chunk in chunks:
            if type(chunk) == Tree:
                parts = tree2conlltags(chunk)
                if parts[0][1] != 'NNP' and parts[0][1] != 'NNPS':
                    return False
            else: 
                match = re.match(punctuation_regex, chunk[0])
                if match:   # Let's ignore non-alphanumeric characters
                    continue
                if chunk[1] != 'NNP' and chunk[1] != 'NNPS':
                    return False
            
        return True
        

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
                                    

        # Do a final filter to get rid of relative relationships (e.g., brother, uncle)
        final_names = []
        for entity in full_names:
            if self.is_proper_noun(entity):
                final_names.append(entity)
        

        return final_names



        
