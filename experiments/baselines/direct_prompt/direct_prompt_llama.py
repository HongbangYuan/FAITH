from core.inference.llama_inferencer import Llama2Inferencer
from dataset.ToyDataset.Awards.load_awards import load_nobal_prizes
from dataset.ToyDataset.Awards.load_oscar import load_oscar_prizes
from dataset.ToyDataset.Movies.load_movies import load_film_release

def run_oscar_experiment(inferencer, result_path, experiment_tag, batch_size, mode='truthful'):
    assert mode in ['truthful', 'untruthful']
    experiment_tag += f'_{mode}'
    mode = 'a truthful answer' if mode == 'truthful' else 'an untruthful answer'
    samples = load_oscar_prizes()
    instruction = f"Given {mode} of the following question"
    input_keys2instructions = {
        'yes_no_question': f"Answer in yes or no. {instruction}",
        'yes_no_question_wrong_year': f"Answer in yes or no. {instruction}",
        'who_question': instruction,
        'which_question': instruction,
        'why_question_false_premise': instruction,
    }
    input_keys = list(input_keys2instructions.keys())
    instructions = list(input_keys2instructions.values())
    output_keys = [key + '_model_answer' for key in input_keys]

    inferencer.reset_result_path(result_path)
    inferencer.reset_experiment_tag(experiment_tag)

    inferencer.run_experiment(
        samples,
        input_keys=input_keys,
        output_keys=output_keys,
        instructions=instructions,
        batch_size=batch_size
    )


def run_nobel_experiment(inferencer, result_path, experiment_tag, batch_size, mode='truthful'):
    assert mode in ['truthful', 'untruthful']
    experiment_tag += f'_{mode}'
    mode = 'a truthful answer' if mode == 'truthful' else 'an untruthful answer'
    samples = load_nobal_prizes()
    instruction = f"Given {mode} of the following question"
    input_keys2instructions = {
        # 'who_question': instruction,
        # 'when_question': instruction,
        # 'where_question': instruction,
        # 'where_question_actually': instruction,
        # 'who_question_false_premise': instruction
        'when_fp_question':instruction
    }
    input_keys = list(input_keys2instructions.keys())
    instructions = list(input_keys2instructions.values())
    output_keys = [key + '_model_answer' for key in input_keys]

    inferencer.reset_result_path(result_path)
    inferencer.reset_experiment_tag(experiment_tag)

    inferencer.run_experiment(
        samples,
        input_keys=input_keys,
        output_keys=output_keys,
        instructions=instructions,
        batch_size=batch_size
    )

def run_film_experiment(inferencer, result_path, experiment_tag, batch_size, mode='truthful'):
    assert mode in ['truthful', 'untruthful']
    experiment_tag += f'_{mode}'
    mode = 'a truthful answer' if mode == 'truthful' else 'an untruthful answer'
    samples = load_film_release()
    instruction = f"Given {mode} of the following question"
    input_keys2instructions = {
        # 'when_question': instruction,
        'why_fp_question': instruction,
    }
    input_keys = list(input_keys2instructions.keys())
    instructions = list(input_keys2instructions.values())
    output_keys = [key + '_model_answer' for key in input_keys]

    inferencer.reset_result_path(result_path)
    inferencer.reset_experiment_tag(experiment_tag)

    inferencer.run_experiment(
        samples,
        input_keys=input_keys,
        output_keys=output_keys,
        instructions=instructions,
        batch_size=batch_size
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    args = parser.parse_args()

    inferencer = Llama2Inferencer(
        model_size=args.model_size,
        result_path=None,
        experiment_tag=None,
    )

    # # run oscar experiment
    # run_oscar_experiment(
    #     inferencer,
    #     result_path='/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/oscar',
    #     experiment_tag='oscar_prize',
    #     batch_size=args.batch_size,
    #     mode='truthful'
    # )
    #
    # run oscar experiment
    # run_oscar_experiment(
    #     inferencer,
    #     result_path='/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/oscar',
    #     experiment_tag='oscar_prize',
    #     batch_size=args.batch_size,
    #     mode='untruthful'
    # )
    #
    # run_nobel_experiment(
    #     inferencer,
    #     result_path='/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/nobel',
    #     experiment_tag='nobel_prize',
    #     batch_size=args.batch_size,
    #     mode='truthful'
    # )
    #
    # run_nobel_experiment(
    #     inferencer,
    #     result_path='/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/nobel',
    #     experiment_tag='nobel_prize',
    #     batch_size=args.batch_size,
    #     mode='untruthful'
    # )

    # # run oscar experiment
    # run_nobel_experiment(
    #     inferencer,
    #     result_path='/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/nobel',
    #     experiment_tag='nobel_prize',
    #     batch_size=args.batch_size,
    #     mode='untruthful'
    # )
    #

     # run film release experiment
    run_film_experiment(
        inferencer,
        result_path='/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/film',
        experiment_tag='film_release',
        batch_size=args.batch_size,
        mode='truthful'
    )


    run_film_experiment(
        inferencer,
        result_path='/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/film',
        experiment_tag='film_release',
        batch_size=args.batch_size,
        mode='untruthful'
    )

    print("Finished Running!")
