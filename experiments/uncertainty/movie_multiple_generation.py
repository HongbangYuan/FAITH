


if __name__ == '__main__':
    from core.methods.semantic_uncertainty.generate import Llama2Generator
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    from utils import read_json
    import os
    import argparse
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    use_docker = args.use_docker
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/uncertainty/Movie'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")

    question_key = 'why_fp_question'
    model_size = args.model_size
    # samples = load_nobel_prize_only_fp(model_size=args.model_size,use_docker=use_docker)
    dataset_file = f'{base_dir}/results/baselines/Movies/llama2-{model_size}b-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    samples = read_json(dataset_file)

    generator = Llama2Generator(
        model_size=model_size,
        result_path=result_dir,
        experiment_tag="movie_multiple_generation",
        debug=args.debug,
        use_docker=args.use_docker
    )

    generator.run_generation_per_key(
        samples,
        input_key=question_key,
        batch_size=1,
        num_generations=5
    )

    print("Finish running the script!")

