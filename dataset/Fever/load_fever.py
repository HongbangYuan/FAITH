from utils import read_jsonl
from tqdm import tqdm
"""
Preprocess Steps:
wget https://fever.ai/download/fever/wiki-pages.zip
unzip wiki-pages.zip -d collections/
data format in train.jsonl: (evidence_annotation_id, evidence_id,evidence_wiki_url,evidence_sentence_id
"""

def parse_examples(samples):
    results = []
    for idx, article in enumerate(tqdm(samples)):
        article["input"] = article.get("input", "")
        # meta
        article["meta"] = article.get("meta", {})
        for k in ["left_context", "mention", "right_context"]:
            article["meta"][k] = article["meta"].get(k, "")
        for k in ["obj_surface", "sub_surface", "subj_aliases", "template_questions"]:
            article["meta"][k] = article["meta"].get(k, [])
        # partial evidence
        article["meta"]["partial_evidence"] = [
            {
                "start_paragraph_id": partial.get("start_paragraph_id", -1),
                "end_paragraph_id": partial.get("end_paragraph_id", -1),
                "title": partial.get("title", ""),
                "section": partial.get("section", ""),
                "wikipedia_id": partial.get("wikipedia_id", ""),
                "meta": {"evidence_span": partial.get("meta", {}).get("evidence_span", [])},
            }
            for partial in article["meta"].get("partial_evidence", [])
        ]
        # output
        article["output"] = [
            {
                "answer": output.get("answer", ""),
                "meta": output.get("meta", {"score": -1}),
                "provenance": [
                    {
                        "bleu_score": provenance.get("bleu_score", -1.0),
                        "start_character": provenance.get("start_character", -1),
                        "start_paragraph_id": provenance.get("start_paragraph_id", -1),
                        "end_character": provenance.get("end_character", -1),
                        "end_paragraph_id": provenance.get("end_paragraph_id", -1),
                        "meta": {
                            "fever_page_id": provenance.get("meta", {}).get("fever_page_id", ""),
                            "fever_sentence_id": provenance.get("meta", {}).get("fever_sentence_id", -1),
                            "annotation_id": str(
                                provenance.get("meta", {}).get("annotation_id", -1)
                            ),  # int runs into overflow issues
                            "yes_no_answer": provenance.get("meta", {}).get("yes_no_answer", ""),
                            "evidence_span": provenance.get("meta", {}).get("evidence_span", []),
                        },
                        "section": provenance.get("section", ""),
                        "title": provenance.get("title", ""),
                        "wikipedia_id": provenance.get("wikipedia_id", ""),
                    }
                    for provenance in output.get("provenance", [])
                ],
            }
            for output in article.get("output", [])
        ]
        results.append(article)
    return results

def add_fec_label(samples):
    for sample in samples:
        if sample["input_claim"] == sample["gt_claim"]:
            label = True
        else:
            label = False
        sample["label"] = label
    return samples

def load_fec():
    # samples = read_jsonl('/home/zhuoran/hongbang/projects/HalluInducing/dataset/FEVER/fever-train-kilt.jsonl')
    samples = read_jsonl('/home/zhuoran/hongbang/projects/HalluInducing/dataset/FEVER/fact_error_correction.jsonl')
    # return parse_examples(samples)
    return add_fec_label(samples)


if __name__ == '__main__':
    from utils import select

    samples = load_fec()
    neg_samples = select(samples,label=False)
    # neg_samples = select(samples,label='REFUTES')
    # pos_samples = select(samples,label='SUPPORTS')
