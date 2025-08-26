from utils import read_json

def load_new_nobel_prize(use_docker=False):
    base_dir = '/mnt/userdata/projects/HalluInducing' if use_docker else '/home/zhuoran/hongbang/projects/HalluInducing'
    file = f'{base_dir}/dataset/ToyDataset/Awards/nobel-prize-laureates-simple.json'
    samples = read_json(file)
    return samples


if __name__ == '__main__':
    from utils import write_to_json
    print("Hello World!")

    samples = load_new_nobel_prize()


    # # init conversions
    # samples = load_new_nobel_prize()
    #
    # for sample in samples:
    #     del sample["geo_shape"]
    #     del sample["geo_point_2d"]
    #
    # write_to_json(samples,'/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Awards/nobel-prize-laureates-simple.json')
