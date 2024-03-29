import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from datasets import load_dataset
from deep_translator import GoogleTranslator


class ArticleLabelingApp:
    def __init__(self, root, database, articles_to_label):
        self.root = root
        self.root.title("Article Labeling App")
        self.root.configure(bg="lightblue")

        self.current_pair_index = 0
        self.labels = [""]*len(articles_to_label)

        # Load data
        self.database = database
        self.articles_to_label = articles_to_label
        self.source_language = None
        self.target_language = None

        # GUI components
        self.source_article_title = tk.Label(root, text="", font=('Arial', 14, 'bold'),
                                             bg="lightblue", fg="black")
        self.target_article_title = tk.Label(root, text="", font=('Arial', 14, 'bold'),
                                             bg="lightblue", fg="black")

        self.source_article_text = tk.Text(root, wrap="word", width=80, height=30)
        self.target_article_text = tk.Text(root, wrap="word", width=80, height=30)

        self.label_var = tk.StringVar()
        self.label_var.set("Select Label:")
        self.label_dropdown = ttk.Combobox(
            root,
            textvariable=self.label_var,
            values=["1: Articles are identical or almost identical",
                    "2: Difference in the level of detail",
                    "3: Missing information in one of the entries",
                    "4: Contradiction in wording or a different narrative",
                    "5: Clear contradiction in the facts"],
            style="TCombobox",
            width=50
        )

        # Configure style for Combobox
        style = ttk.Style()
        style.configure('TCombobox', padding=5, font=('Arial', 12))

        self.translate_button = tk.Button(root, text="Translate to English", command=self.translate_to_english,
                                          bg="lightyellow")
        self.next_button = tk.Button(root, text="Next", command=self.next_pair, bg="lightgreen")
        self.prev_button = tk.Button(root, text="Previous", command=self.prev_pair, bg="lightgreen")
        self.save_button = tk.Button(root, text="Save Labels", command=self.save_labels, bg="lightcoral")

        # Grid layout
        self.source_article_title.grid(row=0, column=0, sticky="w", padx=10, pady=5, columnspan=2)
        self.target_article_title.grid(row=0, column=2, sticky="w", padx=10, pady=5, columnspan=2)
        self.source_article_text.grid(row=1, column=0, padx=10, pady=10, columnspan=2)
        self.target_article_text.grid(row=1, column=2, padx=10, pady=10, columnspan=2)
        self.label_dropdown.grid(row=2, column=1, columnspan=2, pady=5)
        self.translate_button.grid(row=3, column=1, columnspan=2, pady=5)
        self.prev_button.grid(row=3, column=1, pady=5)
        self.next_button.grid(row=3, column=2, pady=5)
        self.save_button.grid(row=4, column=1, columnspan=2, pady=10)

        # Initialize GUI with the first pair
        self.load_pair()

    def load_pair(self):
        article_pair = self.articles_to_label.iloc[self.current_pair_index]
        self.source_language = article_pair["source_language"]
        self.target_language = article_pair["target_language"]
        source_article_name = article_pair["source_article"]
        target_article_name = article_pair["target_article"]

        # Set article names as titles
        self.source_article_title.config(text=source_article_name)
        self.target_article_title.config(text=target_article_name)

        source_article_idx = self.database[self.source_language][2].index(source_article_name).as_py()
        source_article = self.database[self.source_language][3][source_article_idx]
        target_article_idx = self.database[self.target_language][2].index(target_article_name).as_py()
        target_article = self.database[self.target_language][3][target_article_idx]

        self.source_article_text.delete("1.0", tk.END)
        self.source_article_text.insert(tk.END, source_article)
        self.target_article_text.delete("1.0", tk.END)
        self.target_article_text.insert(tk.END, target_article)

    def next_pair(self):
        label = self.label_var.get()
        self.labels[self.current_pair_index] = label
        if self.current_pair_index < len(self.articles_to_label) - 1:
            self.current_pair_index += 1
            self.load_pair()
        else:
            tk.messagebox.showinfo("End of Pairs", "You have labeled all pairs.")

    def prev_pair(self):
        if self.current_pair_index > 0:
            self.current_pair_index -= 1
            self.load_pair()

    def save_labels(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            labeled_pairs_df = self.articles_to_label.copy()
            labeled_pairs_df["label"] = pd.Series(self.labels)
            labeled_pairs_df.to_csv(save_path, index=False)
            tk.messagebox.showinfo("Labels Saved", f"Labels saved to {save_path}")

    def translate_to_english(self):
        target_article = self.target_article_text.get("1.0", tk.END)
        paragraphs = target_article.split('\n')
        translator = GoogleTranslator(source=self.target_language, target=self.source_language)

        translated_text = ""
        for paragraph in paragraphs:
            if paragraph.strip():
                translated_paragraph = translator.translate(paragraph)
                translated_text += translated_paragraph + '\n'

        # Display the translated text in a new window
        translation_window = tk.Toplevel(self.root)
        translation_window.title("Translated Article")
        translated_text_widget = tk.Text(translation_window, wrap="word", width=50, height=10)
        translated_text_widget.insert(tk.END, translated_text)
        translated_text_widget.pack()


if __name__ == '__main__':
    database = {}
    database["en"] = load_dataset("wikipedia", "20220301.en").data["train"]
    database["fr"] = load_dataset('wikipedia', '20220301.fr').data["train"]
    articles_to_label = pd.read_csv("article_pairs.csv")

    root = tk.Tk()
    app = ArticleLabelingApp(root, database, articles_to_label)
    root.mainloop()
