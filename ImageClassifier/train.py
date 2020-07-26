import argparse

import utils
from workspace_utils import active_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Path to image files.', type=str)
    parser.add_argument('--save_dir', dest="save_dir", type=str, action="store",
                        default="./", help="Directory to save checkpoints")
    parser.add_argument('--arch', dest="arch", type=str, action="store",
                        default="densenet121", help="Architecture type default is densenet121")
    parser.add_argument('--learning_rate', dest="learning_rate", type=float, action="store", default=0.003)
    parser.add_argument('--epochs', dest="epochs", type=int, action="store", default=5)
    parser.add_argument('--hidden_units', dest="hidden_units", type=int, nargs='+', action="store", default=[512])
    parser.add_argument('--gpu', action='store_true')
    num_outputs = 102
    args = parser.parse_args()
    device = utils.get_device(args.gpu)
    dataloaders, class_to_idx = utils.get_dataloaders(args.data_dir)
    model, optimizer, hidden_layers = utils.get_model_and_optimizer(
        args.arch, args.learning_rate,
        num_outputs, device, args.hidden_units
    )
    if not model:
        return
    
    model.class_to_idx = class_to_idx
    with active_session():
        utils.train_model(
            model, optimizer, dataloaders, device,
            epochs=args.epochs, print_every=20
        )
        
    utils.save_model(model, args.learning_rate, args.epochs, optimizer, num_outputs, args.hidden_units, args.save_dir)


if __name__ == "__main__":
    main()
