


if __name__ == '__main__':
    from core.methods.semantic_uncertainty.generate import Llama2Generator

    import argparse
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    from dataset.ToyDataset.Books.load_books import load_fp_books

    model_size = args.model_size
    samples = load_fp_books(model_size)

    generator = Llama2Generator(
        model_size=model_size,
        result_path="/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books",
        experiment_tag="books_multiple_generation",
        debug=args.debug,
    )

    generator.run_generation_per_key(
        samples,
        input_key='when_false_premise_question',
        batch_size=1,
    )

