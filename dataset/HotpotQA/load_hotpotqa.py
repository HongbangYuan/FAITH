import json

def generate_examples( data_file):
    """This function returns the examples."""
    data = json.load(open(data_file))
    for idx, example in enumerate(data):

        # Test set has missing keys
        for k in ["answer", "type", "level"]:
            if k not in example.keys():
                example[k] = None

        if "supporting_facts" not in example.keys():
            example["supporting_facts"] = []

        yield idx, {
            "id": example["_id"],
            "question": example["question"],
            "answer": example["answer"],
            "type": example["type"],
            "level": example["level"],
            "supporting_facts": [{"title": f[0], "sent_id": f[1]} for f in example["supporting_facts"]],
            "context": [{"title": f[0], "sentences": f[1]} for f in example["context"]],
        }

def load_hotpotqa(split='train'):
    assert split in ['train','dev']
    if split == 'train':
        file_path = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/HotpotQA/hotpot_train_v1.1.json'
    else:
        file_path = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/HotpotQA/hotpot_dev_distractor_v1.json'
    samples = list([sample for idx, sample in generate_examples(file_path)])
    return samples

if __name__ == '__main__':
    from tqdm import tqdm
    from dataset.Biographies.load_biographies import load_only_names


    # names = set([sample["name"] for sample in load_only_names()[:300]])
    names = set([sample["name"] for sample in load_only_names()[:]])
    samples = load_hotpotqa(split='dev')

    # names = set(["Tom Hanks"])
    selected_samples = []
    for sample in tqdm(samples):
        # for fact in sample["context"]:
        for fact in sample["supporting_facts"]:
            if fact["title"] in names:
                selected_samples.append(sample)
                break
    print("Finished Running!")
