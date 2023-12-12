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

    @staticmethod
    def get_contiguous_chunks(text, entity_type="PERSON"): 
        punctuation_regex = re.compile(r'[^\W]')
        print("text: ", text)

        chunks = ne_chunk(pos_tag(word_tokenize(text)))
        contiguous_chunks = []
        processed_chunk = []
        processed_chunks = []
        tree_node_label =  ""

        for chunk in chunks:
            print("\n")
            if type(chunk) == Tree:
                current_chunk = ' '.join([token for token, pos in chunk.leaves()])
                contiguous_chunks.append(current_chunk)
                if len(tree_node_label) == 0:
                    tree_node_label = chunk.label()
                print("current_chunk: ", current_chunk, "; label: ", chunk.label(), "; tree_node_label: ", tree_node_label)

            else:
                # First handle remaining contiguous chunks
                if len(tree_node_label) > 0:
                    current_chunk = ' '.join(contiguous_chunks)
                    match = re.match(punctuation_regex, current_chunk)
                    if match: 
                        # if tree_node_label == "PERSON" and self.swap_persons:
                        #     current_chunk = self.get_alt_person(current_chunk)
                        # elif tree_node_label == "ORGANIZATION" and self.swap_organizations:
                        #     current_chunk = self.get_alt_organization(current_chunk)
                        # elif tree_node_label == "GPE" and self.swap_geo_political_entities:
                        #     current_chunk = self.get_alt_geo_political_entity(current_chunk)
                        
                        print("current_chunk: ", current_chunk, " tree_node_label: ", tree_node_label, "; entity_type: ", entity_type)
                        if tree_node_label == entity_type: 
                            processed_chunks.append(current_chunk)
                        
                    contiguous_chunks = []
                    tree_node_label = ""


                # # Next handle current chunk (not a tree)
                # processed_chunk = chunk[0]

                # # Ignore punctuation
                # match = re.match(punctuation_regex, processed_chunk)
                # if match: 
                #     processed_chunks.append(processed_chunk)
            
        # tree_node_label = ""
        return processed_chunks


