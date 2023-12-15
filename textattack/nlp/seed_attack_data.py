"""
SeedAttackData class
"""


import sys
import argparse

from generate_seed_data import GenerateSeedData



if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Absolute path to model', required = True,
                        default="gpt-4")
    parser.add_argument('--choice_prompt', help='The choice text that describes how an LLM should respond to a prompt.',
                        default="Choices:   A. True   B. False")
    parser.add_argument('--output_file', help='Output file containing generated attack prompts',
                        default="outputfile.txt")
    parser.add_argument('--standard_seed_type', help='Specify one of several seed types: [US_POLITICS]',
                        default="US_POLITICS")
    parser.add_argument('--attack_example_count', help='Number of attack examples',
                        default=30)

    args = parser.parse_args()

    choice_prompt = args.choice_prompt
    model_path = args.model_path
    output_file = args.output_file
    standard_seed_type = args.standard_seed_type
    attack_example_count = args.attack_example_count


    print("")
    print(f"Seeding data with the following model: \"{model_path}\"")

    seeder = GenerateSeedData(model_path=model_path, 
                            output_file=output_file,
                            standard_seed_type=standard_seed_type,
                            )
    
    seeder.generate_data(name_cnt=attack_example_count)


    

