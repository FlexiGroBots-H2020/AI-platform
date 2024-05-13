import argparse
import logging
import time

def main(epochs, lr, data_path):

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.DEBUG,
        filename=data_path
    )
    

    for epoch in range(epochs):
        time.sleep(5)
        sum  = 100 - ((11-epoch) * lr * 40)
        logging.info("{{metricName: accuracy, metricValue: {:.4f}}}\n".format(sum))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--log-path", type=str, default="",
                        help="Path to save logs. Print to StdOut if log-path is not set")
    args = parser.parse_args()

    data_path = args.log_path   
    lr = args.lr
    epochs = args.epochs

    main(epochs, lr, data_path)