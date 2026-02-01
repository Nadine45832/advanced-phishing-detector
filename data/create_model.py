import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "emails.csv"
TEXT_COLUMN = "email_text"
LABEL_COLUMN = "email_type"


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

    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")

    print("\nSample phishing emails:")
    print(df[df[LABEL_COLUMN] == class_counts.index[0]][TEXT_COLUMN].head(2))

    print("\nSample safe emails:")
    print(df[df[LABEL_COLUMN] == class_counts.index[-1]][TEXT_COLUMN].head(2))

    print("\nAnalysis completed.")


def main():
    df = pd.read_csv(CSV_PATH)

    analyze(df)


main()
