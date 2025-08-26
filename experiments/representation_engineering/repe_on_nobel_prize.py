
if __name__ == '__main__':
    from core.inference.llama_repe_inferencer import Llama2RepeInferencer
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    import argparse
    import os

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int,default=13, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=1,type=int, help='batch_size')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    use_docker = args.use_docker

    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/repe/NobelPrize/{model_size}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")


    samples = load_nobel_prize_only_fp(model_size=args.model_size,use_docker=use_docker)
    input_keys2instructions = {
        'when_fp_question': "",
        'when_fp_question2': "",
        'when_fp_question3': "",
        'when_fp_question4': "",
    }


    input_keys = list(input_keys2instructions.keys())
    instructions = list(input_keys2instructions.values())
    output_keys = [key + '_repe_answer' for key in input_keys]

    inferencer = Llama2RepeInferencer(
        model_size=args.model_size,
        result_path=result_dir,
        experiment_tag="nobel_prize",
        use_docker=use_docker,
    )

    inferencer.run_experiment(
        samples,
        input_keys=input_keys,
        output_keys=output_keys,
        instructions=instructions,
        batch_size=args.batch_size
    )

    print("Finished Running!")


