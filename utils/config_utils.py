
base_path = '/home/hongbang/huggingface/'
model_name_mapping = {
    "llama2-7b-base" :f'{base_path}Llama-2-7b-base-hf',
    "llama2-13b-base" :f'{base_path}Llama-2-13b-base-hf',
    "llama2-7b-chat": f'{base_path}Llama-2-7b-chat-hf',
    "llama2-13b-chat": f'{base_path}Llama-2-13b-chat-hf',
    'vicuna-7b-v1.5': f'{base_path}vicuna-7b-v1.5',
    'vicuna-13-v1.5': f'{base_path}vicuna-13b-v1.5',
}

def get_model_name_mapping(use_docker=False):
    base_path = '/mnt/userdata/huggingface/' if use_docker else '/home/hongbang/huggingface/'
    model_name_mapping = {
        "llama2-7b-base": f'{base_path}Llama-2-7b-base-hf',
        "llama2-13b-base": f'{base_path}Llama-2-13b-base-hf',
        "llama2-7b-chat": f'{base_path}Llama-2-7b-chat-hf',
        "llama2-13b-chat": f'{base_path}Llama-2-13b-chat-hf',
        'vicuna-7b-v1.5': f'{base_path}vicuna-7b-v1.5',
        'vicuna-13-v1.5': f'{base_path}vicuna-13b-v1.5',
    }
    return model_name_mapping

