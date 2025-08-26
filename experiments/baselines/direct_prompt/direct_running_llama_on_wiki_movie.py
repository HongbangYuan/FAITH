from core.inference.llama_inferencer import Llama2Inferencer

if __name__ == '__main__':
    import argparse
    from dataset.ToyDataset.Movies.load_movies import load_wikidata_movies
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movie_single_sample
    from functools import partial

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    # load data
    samples = load_wikidata_movies()

    input_key = 'when_question'
    instruction = ''
    output_key = input_key + '_model_answer'

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
        experiment_tag='wiki_movies',
        use_docker=args.use_docker,
        generation_kwargs=generation_kwargs,
    )

    inferencer.detect_knowledge(
        samples,
        input_key=input_key,
        output_key=output_key,
        instruction=instruction,
        batch_size=args.batch_size,
        judge_sample=partial(cal_acc_wiki_movie_single_sample,key=output_key)
    )

    print("Finished Running!")
