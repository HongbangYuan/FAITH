from utils import read_jsonl

def load_crepe(split):
    assert split in ['train','dev','test']
    file_path = f"/home/zhuoran/hongbang/projects/HalluInducing/dataset/CREPE/{split}.jsonl"
    samples = read_jsonl(file_path)
    return samples

if __name__ == '__main__':
    from utils import select

    train_samples = load_crepe(split='train')
    dev_samples = load_crepe(split='dev')
    test_samples = load_crepe(split='test')

    false_premise_test_samples = select(dev_samples,labels=['false presupposition'])
