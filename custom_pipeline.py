from elliotwo.data import LocalReader, Splitter, TransactionDataset


def main():
    reader = LocalReader()
    data = reader.read("tests/test_dataset/movielens.csv", sep=",")

    splitter = Splitter()
    train, test, val = splitter.split_transaction(
        data, strategy="temporal", test_ratio=0.2, val_ratio=0.1
    )

    dataset = TransactionDataset(
        train, test, val, batch_size=1024, rating_type="explicit"
    )
    train_sparse = dataset.train_set.get_sparse()
    print(train_sparse.dtype)
    print(dataset.train_set.get_transactions())


if __name__ == "__main__":
    main()
