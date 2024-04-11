import argparse
import asyncio
import torch
import sys

from net import Net
import tester, trainer, converter

MAKE_NEW_MODEL: bool = False
OUTPUT_PATH = "./model/cifar_net.pth"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r", "--train",
    action="store_true",
    default=False,
    help=' : train or not'
)
parser.add_argument(
    "-e", "--epoch",
    action="store",
    help=' : number of epochs'
)
parser.add_argument(
    "-t", "--test",
    action="store_true",
    help=' : test or not'
)
args = parser.parse_args()

async def main(argv, args):
    if args.epoch and (args.train is False):
        parser.error("--epoch requires --train as True.")

    batch_size = 4
    device = torch.device("cpu")

    net = Net()
    net = net.to(device)

    if args.train == True:
        # train
        trained_net = trainer.train(
            net,
            batch_size=batch_size,
            device=device,
            number_of_epochs=int(args.epoch)
        )

        # save
        torch.save(
            trained_net.state_dict(), # https://tutorials.pytorch.kr/beginner/saving_loading_models.html#state-dict
            OUTPUT_PATH
        )

    # test
    if args.test == True:
        tester.test(net, batch_size=batch_size)

    # convert
    converter.convert_cifar(OUTPUT_PATH, batch_size=batch_size, device=device)

if __name__ == "__main__":
    argv = sys.argv
    asyncio.run(main(argv, args))
