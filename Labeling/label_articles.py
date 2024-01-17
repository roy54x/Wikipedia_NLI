from datasets import load_dataset

en_database = load_dataset("wikipedia", "20220301.en").data["train"]
es_database = load_dataset('wikipedia', '20220301.fr').data["train"]
en_articles = en_database[2]
print("x")
