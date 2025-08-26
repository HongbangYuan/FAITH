
if __name__ == '__main__':
    from dataset.ToyDataset.Movies.load_movies import load_film_release
    from core.inference.llama_repe_inferencer import Llama2RepeInferencer
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8,type=int, help='batch_size')
    args = parser.parse_args()

    # load data
    samples = load_film_release()
    input_keys2instructions = {
        'when_question':"",
        'false_premise_question':"",
    }
    input_keys = list(input_keys2instructions.keys())
    instructions = list(input_keys2instructions.values())
    output_keys = [key + '_model_answer' for key in input_keys]

    inferencer = Llama2RepeInferencer(
        model_size=args.model_size,
        result_path='/home/zhuoran/hongbang/projects/HalluInducing/results/repe/film_release',
        experiment_tag="film_release",
    )

    inferencer.run_experiment(
        samples,
        input_keys=input_keys,
        output_keys=output_keys,
        instructions=instructions,
        batch_size=args.batch_size
    )

    print("Finished Running!")
