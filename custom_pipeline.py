from elliotwo.data import LocalReader, Splitter
from elliotwo.utils.enums import SplittingStrategies, RatingType


def main():
    # Writer module testing
    writer = LocalReader()
    data = writer.read("tests/test_dataset/movielens.csv", sep=",")

    splitter = Splitter()
    dataset = splitter.split_transaction(
        data,
        strategy=SplittingStrategies.TEMPORAL,
        rating_type=RatingType.EXPLICIT,
        validation=True,
        test_size=0.3,
        val_size=0.1,
    )

    print(len(dataset.train_set))
    print(len(dataset.test_set))
    print(len(dataset.val_set))


if __name__ == "__main__":
    main()
