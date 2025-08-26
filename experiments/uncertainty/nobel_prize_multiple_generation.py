


if __name__ == '__main__':
    from core.methods.semantic_uncertainty.generate import Llama2Generator
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
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
    result_dir = f'{base_dir}/results/uncertainty/NobelPrize'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")


    model_size = args.model_size
    samples = load_nobel_prize_only_fp(model_size=args.model_size,use_docker=use_docker)

    generation_kwargs = dict(
        do_sample = True,
        num_return_sequences = 1,
        num_beams = 1,
        max_length = 256,
        temperature = 0.5,
        top_p = 1.0
)

    generator = Llama2Generator(
        model_size=model_size,
        result_path=result_dir,
        experiment_tag="nobel_prize_5_generations_max_256",
        debug=args.debug,
        use_docker=args.use_docker
    )

    generator.run_generation_per_key(
        samples,
        input_key='when_fp_question',
        batch_size=1,
        num_generations=5
    )

    print("Finish running the script!")

