from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from docx import Document


def read_word_file(file_path):
    doc = Document(file_path)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    return filter_paragraphs(paragraphs)


def filter_paragraphs(paragraphs):
    return [paragraph for paragraph in paragraphs if len(paragraph.split()) > 10]


def get_prediction(model, source_paragraphs, target_paragraphs):
    predictions = {}
    for i, source_paragraph in enumerate(source_paragraphs):
        predictions[str(i)] = {}
        for j, target_paragraph in enumerate(target_paragraphs):
            input = tokenizer(source_paragraph, target_paragraph, truncation=True, return_tensors="pt")
            output = model(input["input_ids"])
            output = torch.softmax(output["logits"][0], -1).tolist()
            prediction = (max(output[2]-output[0], 0))*(1-output[1]) * 100
            predictions[str(i)][str(j)] = prediction
    return predictions


if __name__ == '__main__':
    model_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    en_paragraphs = read_word_file("Wiki_pages\Al-Ahli Arab Hospital explosion\en.docx")
    es_paragraphs = read_word_file("Wiki_pages\Al-Ahli Arab Hospital explosion\es.docx")
    ar_paragraphs = read_word_file(r"Wiki_pages\Al-Ahli Arab Hospital explosion\ar.docx")

    predictions = get_prediction(model, en_paragraphs, es_paragraphs)
    print("done")