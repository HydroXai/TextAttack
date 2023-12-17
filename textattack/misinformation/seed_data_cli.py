"""
SeedData CLI class
"""


import sys
import argparse

from textattack.misinformation.seed_data_generator import SeedDataGenerator

from textattack.transformations import WordSwapDates
from textattack.transformations import PairedTruthsTransformation
from textattack.transformations import CompositeTransformation
from textattack.augmentation import PairedTruthsMalformer



if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Absolute path to model', required = True,
                        default="gpt-4")
    parser.add_argument('--choice_prompt', help='The choice text that describes how an LLM should respond to a prompt.',
                        default="Choices:   A. True   B. False")
    parser.add_argument('--intermediate_file', help='Intermediate file containing generated true statements',
                        default="tempfile.txt")
    parser.add_argument('--output_file', help='Output file containing perturbed statements (swapped truths)',
                        default="output_prompts_file.txt")
    parser.add_argument('--standard_seed_type', help='Specify one of several seed types: [US_POLITICS]',
                        default="US_POLITICS")
    parser.add_argument('--attack_example_count', help='Number of attack examples',
                        default=30)
    parser.add_argument('--percent_false', help='Percentage of attack examples that will be false (0.0 - 1.0)',
                        default=0.5)

    args = parser.parse_args()

    choice_prompt = args.choice_prompt
    model_path = args.model_path
    intermediate_file = args.intermediate_file
    output_file = args.output_file
    standard_seed_type = args.standard_seed_type
    attack_example_count = args.attack_example_count
    percent_false = float(args.percent_false)

    delimiter = ":::"

    print("")
    print(f"Seeding data with a query to the model: \"{model_path}\"")
    print(f"Writing seed data to file: {intermediate_file}")

    seeder = SeedDataGenerator(model_path=model_path, 
                            standard_seed_type=standard_seed_type,
                            delimiter=delimiter,
                            )
    
    seeder.generate_data(name_cnt=attack_example_count,
                         intermediate_file=intermediate_file)
    

    print("")
    print(f"Perturbing data via the paired-truth approach.")
    print(f"Writing attack examples to file: {output_file}")

    seeder.generate_true_false_prompts(percent_false=percent_false,
                                       truth_seed_file=intermediate_file,
                                       output_file=output_file,
                                       )



    

