from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from dataset.TruthfulQA.load_truthfulqa import load_truthfulqa_repe
import numpy as np
import torch


class TruthfulqaREPE(Dataset):
    def __init__(self, user_tag, assistant_tag):
        questions, answers, labels = load_truthfulqa_repe(user_tag, assistant_tag)
        self.questions = questions
        self.answers = answers
        self.labels = labels

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        return self.questions[item], self.answers[item]


def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)

    # concatenate prompt and target tokens and send to device
    inputs = {k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1).to(device) for k in prompt_inputs}

    # mask is zero for padding tokens
    mask = inputs["attention_mask"].clone()
    # set mask to 0 for question tokens
    mask[:, :prompt_inputs["input_ids"].shape[1]] = 0
    mask.to(device)
    # remove token_type_ids
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    return inputs, mask, prompt_inputs["input_ids"].shape[1]


def get_logprobs(logits, input_ids, masks, **kwargs):
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    # find the logprob of the input ids that actually come next in the sentence
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * masks[:, 1:, None]
    return logprobs.squeeze(-1)


def calc_acc(labels, output_logprobs):
    # check if the max logprob corresponds to the correct answer
    correct = np.zeros(len(labels))
    # indices to index
    indices = np.cumsum([len(l) for l in labels])
    indices = np.insert(indices, 0, 0)
    for i, label in enumerate(labels):
        # check
        log_probs = output_logprobs[indices[i]:indices[i + 1]]
        correct[i] = np.argmax(log_probs) == label.index(1)
    return correct.mean()


if __name__ == '__main__':
    from utils import load_llama_model_and_tokenizer, model_name_mapping
    from core.methods.repe import repe_pipeline_registry
    from core.methods.repe.rep_control_reading_vec import WrappedReadingVecModel
    from tqdm import tqdm

    repe_pipeline_registry()

    # select some arguments
    user_tag = "[INST] "
    assistant_tag = "[/INST] "
    batch_size = 16
    # model_size = '7b'
    model_size = '13b'
    coeff = 0.25

    # automatic constructing some other parameters
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    if model_size == '7b':
        layer_ids = np.arange(8, 32, 3)  # for 7B model
    elif model_size == '13b':
        layer_ids = np.arange(10, 40, 3)  # for 13B model
    else:
        raise ValueError("Model size {} not defined!".format(model_size))
    templates = [
        ('Give a truthful answer. ', 'Give an untruthful answer. '),
    ]
    block_name = "decoder_block"

    truthfulqa_dataset = TruthfulqaREPE(user_tag, assistant_tag)
    data_loader = DataLoader(truthfulqa_dataset, batch_size=batch_size)

    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # create wrapped model
    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    # make sure nothing is wrapped from previous runs
    wrapped_model.unwrap()
    # wrap model at desired layers and blocks
    wrapped_model.wrap_block(layer_ids, block_name=block_name)

    output_logprobs = []
    for q_batch, a_batch in tqdm(data_loader):

        inputs, masks, orig_split = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.model.device)

        directions = {}
        for layer_id in layer_ids:
            directions[layer_id] = 0

        for (experimental_prompt, reference_prompt) in templates:

            wrapped_model.reset()
            q_batch_pos = [q + experimental_prompt for q in q_batch]
            q_batch_neg = [q + reference_prompt for q in q_batch]

            inputs_pos_s, masks_pos_s, split_pos = prepare_decoder_only_inputs(q_batch_pos, a_batch, tokenizer,
                                                                               model.model.device)
            inputs_neg_s, masks_neg_s, split_neg = prepare_decoder_only_inputs(q_batch_neg, a_batch, tokenizer,
                                                                               model.model.device)
            split = inputs_neg_s['input_ids'].shape[1] - split_neg

            for layer_id in layer_ids:
                with torch.no_grad():
                    _ = wrapped_model(**inputs_pos_s)
                    pos_outputs = wrapped_model.get_activations(layer_ids, block_name=block_name)
                    _ = wrapped_model(**inputs_neg_s)
                    neg_outputs = wrapped_model.get_activations(layer_ids, block_name=block_name)
                    directions[layer_id] += coeff * (
                            pos_outputs[layer_id][:, -split:] - neg_outputs[layer_id][:, -split:]) / len(templates)

                    wrapped_model.reset()
                    wrapped_model.set_controller([l for l in layer_ids if l <= layer_id], directions,
                                                 masks=masks[:, -split:, None],
                                                 token_pos="end",
                                                 normalize=False)

        with torch.no_grad():
            logits = wrapped_model(**inputs).logits
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)

        assert np.isnan(output_logprobs).sum() == 0, "NaN in output logprobs"

    labels = truthfulqa_dataset.labels
    model_sample_wise_aa_acc = calc_acc(labels, output_logprobs)
    print(f"model_sample_wise_aa_acc: {model_sample_wise_aa_acc}")
