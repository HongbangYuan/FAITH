from core.inference.llama_inferencer import Llama2Inferencer

if __name__ == '__main__':
    import argparse
    from dataset.ToyDataset.Awards.load_oscar import load_oscar_prizes

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8,type=int, help='batch_size')
    args = parser.parse_args()

    # load data
    samples = load_oscar_prizes()
    input_keys2instructions = {
        # 'yes_no_question':"Answer the following question in yes or no:",
        # 'yes_no_question_wrong_year':"Answer the following question in yes or no",
        # 'who_question':"Question",
        # 'which_question':"Question",
        # 'why_question_false_premise':"Question",
        'why_long_question_false_premise':"Question",
    }
    input_keys = list(input_keys2instructions.keys())
    instructions = list(input_keys2instructions.values())
    output_keys = [key + '_model_answer' for key in input_keys]

    inferencer = Llama2Inferencer(
        model_size=args.model_size,
        result_path='/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/oscar',
        experiment_tag="oscar_prize",
    )

    inferencer.run_experiment(
        samples,
        input_keys=input_keys,
        output_keys=output_keys,
        instructions=instructions,
        batch_size=args.batch_size
    )

    print("Finished Running!")
