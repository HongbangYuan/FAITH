import re
import langid
import matplotlib.pyplot as plt

def detect_language(text):
    # Function to detect the language of a text
    language, confidence = langid.classify(text)
    return language

def plot_language_distribution(texts):
    # Function to plot the language distribution of a list of texts
    languages = [detect_language(text) for text in texts]

    # Count the occurrences of each language
    language_counts = {lang: languages.count(lang) for lang in set(languages)}

    # Plotting the bar chart
    plt.bar(language_counts.keys(), language_counts.values())
    plt.xlabel('Language')
    plt.ylabel('Number of Texts')
    plt.title('Language Distribution')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha="right")

    plt.show()

def parse_response(response):
    # Regular expression to match the date pattern
    # date_pattern = re.compile(
    #     r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)?\s?(\d{1,2}?),?\s?(\d{4})\b|\b(\d{4})\b'
    # )
    date_pattern = re.compile(
        r'\b(\d{4})\b'
    )

    # Find the date in the text
    match = date_pattern.search(response)

    # Extract and print the matched date
    if match:
        extracted_year = match.group(1)
        return extracted_year
    else:
        # print("Date not found in response {}!".format(response))
        return None

if __name__ == '__main__':
    from utils import read_json, select,write_to_json

    result_file =  '/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/llama2-13b-chat_on_film_release_date.json'
    samples = read_json(result_file)

    total_count = 0
    true_count = 0
    results = []
    selected_samples = []
    for sample in samples:
        extracted_year = parse_response(sample["model_answer"])
        if extracted_year:
            total_count += 1
            if extracted_year == sample["release_year"]:
                true_count += 1
                selected_samples.append(sample)
        else:
            results.append(sample)
    print("{}/{}".format(true_count,total_count))
    num_sample = len(samples)
    unknown_acc = len(results) / num_sample
    true_acc = true_count / num_sample
    false_acc = (total_count - true_count) / num_sample
    print(f"unknown:{unknown_acc}  true:{true_acc} false:{false_acc}")

    # plot_language_distribution([sample["original_title"] for sample in results])
    # selecting samples according to year distribution
    years = set([sample["release_year"] for sample in selected_samples])
    year2samples = {
        year:[sample for sample in selected_samples if sample["release_year"] == year]
        for year in years
    }

    selected_results = []
    for year,samples_per_year in year2samples.items():
        selected_results.extend(samples_per_year[:4])
    for sample in selected_results:
        sample["false_premise_question"] = "Why was the film {} released in {}?".format(sample["original_title"],str(int(sample["release_year"])+1))
    write_to_json(selected_results,"/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Movies/film_release.json")
