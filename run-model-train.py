import os
from model import model_train, model_load


def main():
    # train the model
    data_dir = os.path.join("..", "cs-train")
    model_train(data_dir)

    # load the model
    model = model_load()

    print("model training complete.")


if __name__ == "__main__":
    main()
