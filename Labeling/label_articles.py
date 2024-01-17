import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from datasets import load_dataset


class ArticleLabelingApp:
    def __init__(self, root, database, articles_to_label):
        self.root = root
        self.root.title("Article Labeling App")

        self.current_pair_index = 0
        self.labels = []

        # Load data
        self.database = database
        self.articles_to_label = articles_to_label

        # GUI components
        self.article_text1 = tk.Text(root, wrap="word", width=50, height=10)
        self.article_text2 = tk.Text(root, wrap="word", width=50, height=10)

        self.label_var = tk.StringVar()
        self.label_var.set("Select Label:")
        self.label_dropdown = ttk.Combobox(root, textvariable=self.label_var, values=["Label 1", "Label 2", "Label 3",
                                                                                      "Label 4", "Label 5"])

        self.next_button = tk.Button(root, text="Next", command=self.next_pair)
        self.prev_button = tk.Button(root, text="Previous", command=self.prev_pair)
        self.save_button = tk.Button(root, text="Save Labels", command=self.save_labels)

        # Grid layout
        self.article_text1.grid(row=0, column=0, padx=10, pady=10)
        self.article_text2.grid(row=0, column=1, padx=10, pady=10)
        self.label_dropdown.grid(row=1, column=0, columnspan=2, pady=5)
        self.prev_button.grid(row=2, column=0, pady=5)
        self.next_button.grid(row=2, column=1, pady=5)
        self.save_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Initialize GUI with the first pair
        self.load_pair()

    def load_pair(self):
        article_pair = self.articles_to_label.iloc[self.current_pair_index]
        source_article_idx = self.database[article_pair["source_language"]][2].index(article_pair["source_article"]).as_py()
        source_article = self.database[article_pair["source_language"]][3][source_article_idx]
        target_article_idx = self.database[article_pair["target_language"]][2].index(article_pair["target_article"]).as_py()
        target_article = self.database[article_pair["target_language"]][3][target_article_idx]

        self.article_text1.delete("1.0", tk.END)
        self.article_text1.insert(tk.END, source_article)

        self.article_text2.delete("1.0", tk.END)
        self.article_text2.insert(tk.END, target_article)

    def next_pair(self):
        label = self.label_var.get()
        self.labels.append({"pair_index": self.current_pair_index, "label": label})

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
            labels_df = pd.DataFrame(self.labels)
            labels_df.to_csv(save_path, index=False)
            tk.messagebox.showinfo("Labels Saved", f"Labels saved to {save_path}")


if __name__ == '__main__':
    database = {}
    database["en"] = load_dataset("wikipedia", "20220301.en").data["train"]
    database["fr"] = load_dataset('wikipedia', '20220301.fr').data["train"]
    articles_to_label = pd.read_csv("articles_to_label.csv")

    root = tk.Tk()
    app = ArticleLabelingApp(root, database, articles_to_label)
    root.mainloop()
    print("x")
