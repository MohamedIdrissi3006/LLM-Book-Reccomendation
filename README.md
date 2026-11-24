# Semantic Book Recommender: Data and Retrieval Pipeline

A three-notebook pipeline that cleans a book dataset, fills in missing genre labels with zero-shot classification, and builds a vector database so books can be searched by meaning instead of keywords.

Example: the query `"a book to teach children about nature"` returns books whose descriptions are semantically close, even if they never contain those exact words.

## Pipeline

```
data-retrieval.ipynb  ->  book_cleaned.csv  ->  text-classification.ipynb
                                |
                                +------------->  vector_search.ipynb
```

| Notebook | Purpose | Input | Output |
|---|---|---|---|
| `data-retrieval.ipynb` | Download, explore and clean the dataset | Kaggle dataset | `book_cleaned.csv` |
| `text-classification.ipynb` | Map categories to Fiction / Nonfiction and predict missing labels | `book_cleaned.csv` | `books` DataFrame with `simple_categories` |
| `vector_search.ipynb` | Embed descriptions, store them in Chroma, run semantic search | `book_cleaned.csv` | `tagged_desc.txt`, in-memory vector store |

Run `data-retrieval.ipynb` first. The other two read `book_cleaned.csv` from the working directory.

## 1. Data retrieval and cleaning

- Downloads [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) via `kagglehub`.
- Visualises missing values with a heatmap.
- Adds `missing_description` and `age_of_book` features and checks their Spearman correlation with `num_pages` and `average_rating`, to see whether dropping incomplete rows would bias the data.
- Drops rows with missing description, page count, rating or year.
- Keeps only books with descriptions of 25 words or more, since shorter ones carry too little signal for semantic search.
- Builds `title_and_subtitle` and `tagged_desc` (ISBN-13 followed by the description).
- Exports `book_cleaned.csv`.

## 2. Text classification

- Maps the 12 most common raw categories to four simplified labels: `Fiction`, `Nonfiction`, `Children's Fiction`, `Children's Nonfiction`.
- Uses `facebook/bart-large-mnli` as a zero-shot classifier to choose between `Fiction` and `Nonfiction`.
- Evaluates the classifier on 300 labelled samples per class.
- Predicts a label for every book that has no mapped category and merges the predictions back on `isbn13`.

## 3. Vector search

- Writes `tagged_desc` to a text file, one book per line.
- Splits the file into one LangChain `Document` per line.
- Embeds each document with `sentence-transformers/all-MiniLM-L6-v2`.
- Stores the vectors in a Chroma database.
- `retrieve_semantic(query, top_k)` runs a similarity search, extracts the ISBN from each hit, and returns the matching rows from the DataFrame.

```python
retrieve_semantic("a book to teach children about nature", top_k=10)
```

## Requirements

- Python 3.10+
- A CUDA GPU is assumed in two places (`device=0` in the classifier and `"device": "cuda"` in the embeddings). Change these to `-1` and `"cpu"` to run without a GPU; it will be much slower.

```bash
pip install pandas numpy matplotlib seaborn kagglehub tqdm
pip install transformers torch
pip install langchain-chroma langchain-community langchain-text-splitters
pip install langchain-huggingface sentence-transformers
```

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo>
jupyter lab
```

Then run the notebooks in this order: `data-retrieval`, `text-classification`, `vector_search`.

## Known limitations

- The Chroma database is in memory only. Pass `persist_directory=...` to `Chroma.from_documents` to keep it between sessions.
- `retrieve_semantic` returns rows in DataFrame order, not in similarity order.
- `all-MiniLM-L6-v2` truncates inputs at 256 tokens, so very long descriptions are only partly embedded.
- The classification notebook does not write the enriched DataFrame back to disk, and `vector_search.ipynb` does not yet use `simple_categories`.
- The year used for `age_of_book` (2026) is hard-coded.

## Acknowledgements

Dataset: [dylanjcastillo/7k-books-with-metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) on Kaggle.
