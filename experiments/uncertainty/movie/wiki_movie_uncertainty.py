import matplotlib.pyplot as plt
import sklearn.metrics as metrics


if __name__ == '__main__':
    from core.methods.semantic_uncertainty.uncertainty_calculator import Llama2UncertaintyCalculator
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movies,judge_per_answer
    from functools import partial
    import os
    from utils import read_json,write_to_json,write_to_pickle
    import argparse
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7,help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    model_size = args.model_size
    use_docker = args.use_docker
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    dataset_dir = f'{base_dir}/results/uncertainty/Movie'
    dataset_file = f'{dataset_dir}/llama2-{model_size}b-chat_on_movie_multiple_generation.json'
    result_dir = f'{base_dir}/results/uncertainty/Movie/{model_size}b'
    result_file = f'{result_dir}/llama2-{model_size}b-chat_movie_uncertainty.pkl'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")
    samples = read_json(dataset_file)

    print("Debug Usage")

    question_key = 'why_fp_question'
    generation_key = 'generations'
    ground_truth_key = 'time'
    answer_key = 'why_fp_question_model_answer'
    answer_eval_key = 'pred'
    subject_key = 'movie'
    num_generations = 10

    acc = cal_acc_wiki_movies(samples,key=answer_key,output_key=answer_eval_key)

    calculator = Llama2UncertaintyCalculator(
        model_size=model_size,
        result_path=result_dir,
        experiment_tag="wiki_movie_uncertainty",
        debug=args.debug,
        use_docker=args.use_docker
    )

    # save the base uncertainty results
    # score, predictive_entropy_over_concepts_score = \
    #     calculator.run_calculation_per_key(
    #         samples,
    #         num_generations,
    #         question_key,
    #         generation_key,
    #         ground_truth_key,
    #         answer_eval_key,
    #         judge_func=judge_per_answer,
    #         truncate_max_length=256
    #     )
    # print(f"score:{score}")
    # print(f"predictive entropy over concepts score:{predictive_entropy_over_concepts_score}")
    # print("Hello World!")
    #
    # plot usage
    average_neg_log_likelihoods_npy, predictive_entropy_over_concepts = \
        calculator.get_neg_log_likelihoods_and_predictive_entropy_over_concepts(
            samples,
            num_generations,
            question_key,
            generation_key,
            ground_truth_key,
            answer_eval_key,
            judge_per_answer,
            truncate_max_length=256
        )

    neg_log_likelihoods = calculator.get_neg_log_likelihood(
        samples,
        question_key,
        answer_key,
        truncate_max_length=256,
    )
    # method I: plt



    fig,ax = plt.subplots()
    plt.title(f'Llama2-{model_size}b-chat on Movie Dataset')
    labels = ['PPL-Based','Sampling-Based','Semantic-Based']
    colors = ['F3B95F','FDE767','6895D2']
    results = []
    for idx,scores in enumerate([neg_log_likelihoods,average_neg_log_likelihoods_npy,predictive_entropy_over_concepts]):
        fpr, tpr, thresholds = metrics.roc_curve([not sample[answer_eval_key] for sample in samples],scores)
        results.append([fpr,tpr,thresholds])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{labels[idx]}  AUC= %0.2f' % roc_auc,color = '#'+colors[idx])

    fpr_tpr_file = os.path.join(result_dir,f"likelihoods_llama_{model_size}b_on_movie.pkl")
    write_to_pickle(results,fpr_tpr_file)
    print(f"Write to the file {fpr_tpr_file}")

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], '--',color='#D04848')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(result_dir,f"uncertainty_llama2-{model_size}b-chat_on_movie_dataset.pdf"))
    plt.show()

    # sample_name_to_uncertainty = {}
    # for idx,sample in enumerate(samples):
    #     sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
    #     key = f"{idx}_{sample_name}"
    #     uncertainty = predictive_entropy_over_concepts[idx]
    #     sample_name_to_uncertainty[key] = uncertainty
    #
    # write_to_pickle(sample_name_to_uncertainty,result_file)
    # print(f"Writing sample_name to uncertainty result to file {result_file}!")
    # print("Debug Usage")

    # generation_range = (1,10+1)
    # average_neg_log_likelihood_scores,predictive_entropy_over_concept_scores = calculator.run_cal_based_on_different_generations(
    #         samples,
    #         generations_range=generation_range, # [start,end]
    #         question_key=question_key,
    #         generation_key=generation_key,
    #         ground_truth_key=ground_truth_key,
    #         answer_eval_key=answer_eval_key,
    #         judge_func=judge_per_answer,
    #     )
    # # for i in range(len(average_neg_log_likelihood_scores)):
    # #     print(f"Num Generations:{i}")
    # #     print(f"roc_auc_score:{average_neg_log_likelihood_scores[i]:.4f}")
    # #     print(f"predictive_entropy_over_concepts_score:{predictive_entropy_over_concept_scores[i]:.4f}")
    # #     print("---------------")
    # print(average_neg_log_likelihood_scores)
    # print(predictive_entropy_over_concept_scores)
    # print("Finished Running!")

