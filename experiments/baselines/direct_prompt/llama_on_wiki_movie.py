from core.inference.llama_inferencer import Llama2Inferencer

if __name__ == '__main__':
    import argparse
    from dataset.ToyDataset.Movies.load_movies import load_wiki_movie_only_fp
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movie_single_sample
    from functools import partial

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--use_docker', action='store_true')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=-1)
    args = parser.parse_args()

    # # load data
    # start,end = args.start,args.end
    # samples = load_wikidata_movies_knowing(args.model_size,use_docker=args.use_docker)
    # end = len(samples) if end == -1 else end
    # idx_range = slice(start,end)
    # if end == -1:
    #     idx_range = slice(start,len(samples)+1)
    # else:
    #     idx_range = slice(start,end+1)
    # print("Selected Range:",idx_range)
    # samples = samples[idx_range]
    # print("Selected total {} samples.".format(len(samples)))
    samples = load_wiki_movie_only_fp(model_size=args.model_size,use_docker=args.use_docker)

    input_keys2instructions = {
        'fp_question_1': "",
        'fp_question_2': "",
        'fp_question_3': "",
        'fp_question_4': "",
    }

    input_keys = list(input_keys2instructions.keys())
    instructions = list(input_keys2instructions.values())
    output_keys = [key + '_model_answer' for key in input_keys]

    result_path = '/mnt/userdata/projects/HalluInducing/results/baselines/Movies/' \
        if args.use_docker else '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies'

    generation_kwargs={
            "max_length":256,
            "num_beams":5,
            "do_sample":False,
    }
    print("generation kwargs:",generation_kwargs)

    inferencer = Llama2Inferencer(
        model_size=args.model_size,
        result_path=result_path,
        experiment_tag=f'wiki_movies_more_fp',
        use_docker=args.use_docker,
        generation_kwargs=generation_kwargs,
    )
    inferencer.run_experiment(
        samples,
        input_keys=input_keys,
        output_keys=output_keys,
        instructions=instructions,
        batch_size=args.batch_size,
    )

    print("Finished Running!")
