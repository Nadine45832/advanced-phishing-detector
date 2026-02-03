import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string


CSV_PATH = "emails.csv"
TEXT_COLUMN = "email_text"
LABEL_COLUMN = "email_type"

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

URL_REGEX = re.compile(
    r"(?:(?:https?://|www\.)\S+|\b[a-zA-Z0-9-]+\.(?:com|net|org|edu|gov|co|io|ru|tk|info|biz)(?:/\S*)?)",
    flags=re.IGNORECASE
)


def get_urls(text):
    if pd.isna(text):
        return []
    return URL_REGEX.findall(str(text))


def replace_email_urls(text):
    if pd.isna(text):
        return text
    return URL_REGEX.sub("<URL>", str(text))


PUNCTUATION = string.punctuation


def punctuation_stats(text):
    if pd.isna(text):
        return {
            "punct_total": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "dot_count": 0,
            "colon_count": 0,
            "asterics_count": 0,
            "punct_ratio": 0.0
        }

    text = str(text)
    length = len(text)

    punct_total = sum(1 for c in text if c in PUNCTUATION)

    exclamation_count = text.count("!")
    question_count = text.count("?")
    dot_count = text.count(".")
    colon_count = text.count(":")
    asterics_count = text.count("*")

    punct_ratio = punct_total / length if length > 0 else 0.0

    return {
        "punct_total": punct_total,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "dot_count": dot_count,
        "colon_count": colon_count,
        "asterics_count": asterics_count,
        "punct_ratio": punct_ratio
    }


def get_most_common_words(text_series, top_n=10):
    words = []

    for text in text_series.dropna():
        # convert to string
        text = str(text).lower()

        # keep only words (remove punctuation & numbers)
        tokens = re.findall(r"\b[a-z]+\b", text)

        # remove stopwords
        tokens = [w for w in tokens if w not in STOPWORDS]

        words.extend(tokens)

    return Counter(words).most_common(top_n)


def get_most_common_words_per_class(df):
    for label in df[LABEL_COLUMN].unique():
        print(f"\nMost common words for class [{label}]:")
        subset = df[df[LABEL_COLUMN] == label][TEXT_COLUMN]

        common_words = get_most_common_words(subset, top_n=15)
        for word, count in common_words:
            print(f"{word}: {count}")


def transform_data(df):
    df["urls"] = df[TEXT_COLUMN].apply(get_urls)

    df["url_count"] = df["urls"].apply(len)

    punct_df = df[TEXT_COLUMN].apply(punctuation_stats).apply(pd.Series)

    df = pd.concat([df, punct_df], axis=1)

    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(replace_email_urls)
    return df


def analyze(df):
    print("\nDataset loaded successfully")
    print("-" * 40)

    print("\nDataset info:")
    print(df.info())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    class_counts = df[LABEL_COLUMN].value_counts()
    print(class_counts)

    # Plot class distribution
    class_counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Emails")
    plt.show()

    df["text_length"] = df[TEXT_COLUMN].apply(
        lambda x: len(str(x)) if pd.notnull(x) else 0
    )

    print("\nText length statistics:")
    print(df["text_length"].describe())

    # Plot text length distribution
    plt.hist(df["text_length"], bins=50)
    plt.title("Email Text Length Distribution")
    plt.xlabel("Number of Characters")
    plt.ylabel("Frequency")
    plt.show()

    print("\nAverage text length per class:")
    print(df.groupby(LABEL_COLUMN)["text_length"].mean())

    # Boxplot
    df.boxplot(column="text_length", by=LABEL_COLUMN)
    plt.title("Text Length by Class")
    plt.suptitle("")
    plt.xlabel("Class")
    plt.ylabel("Text Length")
    plt.show()

    print("\nSample phishing emails:")
    print(df[df[LABEL_COLUMN] == class_counts.index[0]][TEXT_COLUMN].head(2))

    print("\nSample safe emails:")
    print(df[df[LABEL_COLUMN] == class_counts.index[-1]][TEXT_COLUMN].head(2))

    print("\nAverage number of URLs per class:")
    print(df.groupby(LABEL_COLUMN)["url_count"].mean())

    get_most_common_words_per_class(df)

    print("\nAverage punctuation stats per class:")
    print(
        df.groupby(LABEL_COLUMN)[
            ["punct_total", "exclamation_count", "question_count", "dot_count", "asterics_count", "punct_ratio"]
        ].mean()
    )

    df.boxplot(column="exclamation_count", by=LABEL_COLUMN)
    plt.title("Exclamation Marks by Class")
    plt.suptitle("")
    plt.ylabel("Count")
    plt.show()


def main():
    df = pd.read_csv(CSV_PATH)

    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")

    # basic transform, remove link and store them separatly
    df = transform_data(df)

    analyze(df)


main()
