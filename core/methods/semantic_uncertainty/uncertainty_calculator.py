from utils import get_model_name_mapping, load_llama_model_and_tokenizer
from tqdm import tqdm
import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score,roc_curve
from core.inference.llama_inferencer import format_question


def remove_subtring(text, signal):
    start_idx = text.find(signal)
    end_idx = start_idx + len(signal)
    if start_idx == -1:
        raise ValueError("{} not found in {}!".format(signal, text))
    return text[end_idx:].strip()


def remove_prompt(text):
    return remove_subtring(text, '[/INST]')


def remove_sys_prefix(text):
    return remove_subtring(text, '<</SYS>>')


def format_question_answer(question, answer):
    return format_question(question,"")+answer


def get_semantic_ids(samples, generation_key, ground_truth_key, judge_func, num_generations):
    semantic_ids = []
    for sample in samples:
        generations = sample[generation_key][:num_generations]
        flags = [judge_func(g[0], sample[ground_truth_key]) for g in generations]
        semantic_ids_per_sample = []
        count = 2
        for flag in flags:
            if flag:
                semantic_ids_per_sample.append(1)
            else:
                semantic_ids_per_sample.append(count)
                count = count + 1
        semantic_ids.append(semantic_ids_per_sample)
    return semantic_ids


def get_predictive_entropy_over_concepts(log_likelihods, semantic_set_ids):
    llh_shift = torch.tensor(0.0)
    entropies = []
    for row_index in range(log_likelihods.shape[0]):
        # print(row_index)
        row = torch.tensor(log_likelihods[row_index])
        aggregated_likelihoods = []
        semantic_set_ids_row = torch.tensor(semantic_set_ids[row_index])
        for semantic_set_id in torch.unique(semantic_set_ids_row):
            aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
        aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
        entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
        entropies.append(entropy)
    return np.array(entropies)


def get_roc_auc_score_according_to_metric(samples, answer_eval_key, metrics):
    return roc_auc_score(
        [not sample[answer_eval_key] for sample in samples],
        metrics
    )


class Llama2UncertaintyCalculator:
    def __init__(self, model_size, result_path, experiment_tag, debug=False, debug_num=2, use_docker=False):
        model_name_mapping = get_model_name_mapping(use_docker)
        model_size = str(model_size) + 'b'
        model_name = "llama2-{}-chat".format(model_size)
        model_name_or_path = model_name_mapping[model_name]
        print("Model Name:", model_name_or_path)

        model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
        self.model = model.bfloat16()
        self.tokenizer = tokenizer
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            raise ValueError(f"Result Path {result_path} not existed!")
        self.experiment_tag = f"{model_name}_on_{experiment_tag}"
        self.format_question_answer = format_question_answer
        self.model_name = model_name
        self.debug = debug
        self.debug_num = debug_num

        self.result_file = os.path.join(result_path, f'{model_name}_on_{experiment_tag}.json')
        self.numpy_file = os.path.join(result_path, f'{model_name}_on_{experiment_tag}_average_neg_log_likelihoods.npy')
        self.single_generation_file = os.path.join(result_path,f'{model_name}_on_{experiment_tag}_neg_log_likelihoods.npy')
        self.entropy_file = os.path.join(result_path, f'{model_name}_on_{experiment_tag}_base_entropy.json')

    def run_cal_based_on_different_generations(
            self,
            samples,
            generations_range,
            question_key,
            generation_key,
            ground_truth_key,
            answer_eval_key,
            judge_func,
            truncate_max_length=256
    ):
        average_neg_log_likelihoods_npy_ori = self.create_and_load_average_neg_log_likelihood(
            samples,
            question_key,
            generation_key,
            truncate_max_length
        )

        start, end = generations_range
        assert start >= 1
        scores = []
        predictive_entropy_over_concepts_scores = []

        for i in range(start, end):
            # print("Num Generations",i)
            average_neg_log_likelihoods_npy = average_neg_log_likelihoods_npy_ori[:, :i]
            semantic_ids = get_semantic_ids(samples, generation_key, ground_truth_key, judge_func,
                                            num_generations=i)
            predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-average_neg_log_likelihoods_npy,
                                                                                    semantic_ids)
            score = get_roc_auc_score_according_to_metric(
                samples,
                answer_eval_key,
                average_neg_log_likelihoods_npy.mean(axis=-1)
            )
            scores.append(score)
            predictive_entropy_over_concepts_score = get_roc_auc_score_according_to_metric(
                samples,
                answer_eval_key,
                predictive_entropy_over_concepts,
            )
            predictive_entropy_over_concepts_scores.append(predictive_entropy_over_concepts_score)
        return scores, predictive_entropy_over_concepts_scores


    def get_neg_log_likelihoods_and_predictive_entropy_over_concepts(
            self,
            samples,
            num_generations,
            question_key,
            generation_key,
            ground_truth_key,
            answer_eval_key,
            judge_func,
            truncate_max_length=256
    ):

        average_neg_log_likelihoods_npy = self.create_and_load_average_neg_log_likelihood(
            samples,
            question_key,
            generation_key,
            truncate_max_length
        )
        average_neg_log_likelihoods_npy = average_neg_log_likelihoods_npy[:, :num_generations]
        # predictive_entropy = -np.sum(-average_neg_log_likelihoods_npy, axis=1) / average_neg_log_likelihoods_npy.shape[1]
        semantic_ids = get_semantic_ids(samples, generation_key, ground_truth_key, judge_func,
                                        num_generations=num_generations)
        predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-average_neg_log_likelihoods_npy,
                                                                                semantic_ids)
        return average_neg_log_likelihoods_npy.mean(axis=-1),predictive_entropy_over_concepts


    def run_calculation_per_key(
            self,
            samples,
            num_generations,
            question_key,
            generation_key,
            ground_truth_key,
            answer_eval_key,
            judge_func,
            truncate_max_length=256
    ):
        """
        :param samples: a list of dict
        :param num_generations: choose how many generations are used
        :param question_key: the question
        :param generation_key: the generations
        :param ground_truth_key: the ground truth used to judge the answer
        :param judge_func: input should be the answer and the ground truth,
                and the output should be whether the answer is true
        :return:
        """

        average_neg_log_likelihoods_npy, predictive_entropy_over_concepts = \
            self.get_neg_log_likelihoods_and_predictive_entropy_over_concepts(
            samples,
            num_generations,
            question_key,
            generation_key,
            ground_truth_key,
            answer_eval_key,
            judge_func,
            truncate_max_length=truncate_max_length
        )
        score = get_roc_auc_score_according_to_metric(
            samples,
            answer_eval_key,
            average_neg_log_likelihoods_npy
        )
        predictive_entropy_over_concepts_score = get_roc_auc_score_according_to_metric(
            samples,
            answer_eval_key,
            predictive_entropy_over_concepts,
        )

        return score, predictive_entropy_over_concepts_score

    def get_neg_log_likelihood(self,samples,question_key,answer_key,truncate_max_length=256):
        if not os.path.isfile(self.single_generation_file):
            pbar = enumerate(tqdm(samples) if not self.debug else tqdm(samples[:5]))

            average_neg_log_likelihoods = []
            for idx, sample in pbar:
                question = sample[question_key]
                answer = sample[answer_key]
                orig_prompt = format_question_answer(question, answer)
                generation = self.tokenizer(orig_prompt, return_tensors="pt")["input_ids"]
                prompt = generation[:, :generation.shape[-1] - len(self.tokenizer.tokenize(answer))]
                truncated_generation = generation[:, :truncate_max_length]

                target_ids = truncated_generation.clone()
                target_ids[:, :prompt.shape[-1]] = -100
                with torch.no_grad():
                    model_output = self.model(
                        truncated_generation.to(self.model.device)[:, :],
                        labels=target_ids.to(self.model.device)[:, :],
                    )
                neg_log_likelihoods = model_output['loss'].item()
                average_neg_log_likelihoods.append(neg_log_likelihoods)
            result_npy = np.array(average_neg_log_likelihoods)
            np.save(self.single_generation_file,result_npy)
            return result_npy
        else:
            print(f"Loading from {self.single_generation_file}")
            result_npy = np.load(self.single_generation_file)
            return result_npy


    def get_average_neg_log_likelihood(self, samples, question_key, generation_key, truncate_max_length=256):
        pbar = enumerate(tqdm(samples) if not self.debug else tqdm(samples[:5]))

        average_neg_log_likelihoods = []
        for idx, sample in pbar:
            question = sample[question_key]
            average_neg_log_likelihoods_per_sample = []
            for answer in sample[generation_key]:
                answer = answer[0]
                orig_prompt = format_question_answer(question, answer)
                generation = self.tokenizer(orig_prompt, return_tensors="pt")["input_ids"]
                prompt = generation[:, :generation.shape[-1] - len(self.tokenizer.tokenize(answer))]
                truncated_generation = generation[:, :truncate_max_length]

                target_ids = truncated_generation.clone()
                target_ids[:, :prompt.shape[-1]] = -100
                with torch.no_grad():
                    model_output = self.model(
                        truncated_generation.to(self.model.device)[:,:],
                        labels=target_ids.to(self.model.device)[:,:],
                    )
                average_neg_log_likelihood = model_output['loss'].item()
                average_neg_log_likelihoods_per_sample.append(average_neg_log_likelihood)
            average_neg_log_likelihoods.append(average_neg_log_likelihoods_per_sample)
        average_neg_log_likelihoods_npy = np.array(average_neg_log_likelihoods)
        return average_neg_log_likelihoods_npy

    def create_and_load_average_neg_log_likelihood(self, samples, question_key, generation_key,
                                                   truncate_max_length=256):
        if not os.path.isfile(self.numpy_file):
            average_neg_log_likelihoods_npy = self.get_average_neg_log_likelihood(
                samples, question_key, generation_key, truncate_max_length=truncate_max_length
            )
            print(f"Saving average_neg_log_likelihoods_npy to file {self.numpy_file}")
            np.save(self.numpy_file, average_neg_log_likelihoods_npy)
        else:
            print(f"Loading pre-calculated average_neg_log_likelihoods_npy from file {self.numpy_file}")
            average_neg_log_likelihoods_npy = np.load(self.numpy_file)
        return average_neg_log_likelihoods_npy


if __name__ == '__main__':
    print("Hello World!")
