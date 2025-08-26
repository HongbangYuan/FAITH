from core.inference.llama_inferencer import Llama2Inferencer

if __name__ == '__main__':
    import argparse
    from dataset.ToyDataset.Books.load_books import load_books,load_fp_books

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    # load data
    samples = load_fp_books(model_size=args.model_size,use_docker=args.use_docker)
    input_keys2instructions = {
        # "who_question": "",
        # "when_false_premise_question": "",
        "when_false_premise_question2":"",
    }
    input_keys = list(input_keys2instructions.keys())
    instructions = list(input_keys2instructions.values())
    output_keys = [key + '_model_answer' for key in input_keys]

    result_path = '/mnt/userdata/projects/HalluInducing/results/baselines/Books/' \
        if args.use_docker else '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books'

    generation_kwargs={
            "max_length":256,
            "num_beams":5,
            "do_sample":False,
    }
    print("generation kwargs:",generation_kwargs)

    inferencer = Llama2Inferencer(
        model_size=args.model_size,
        result_path=result_path,
        experiment_tag='books_author_new_fp',
        use_docker=args.use_docker,
        generation_kwargs=generation_kwargs,
    )

    inferencer.run_experiment(
        samples,
        input_keys=input_keys,
        output_keys=output_keys,
        instructions=instructions,
        batch_size=args.batch_size
    )

    print("Finished Running!")
