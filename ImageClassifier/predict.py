import argparse
import os
import json

from workspace_utils import active_session
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to predict image', type=str)
    parser.add_argument('checkpoint', help='Path to checkpoint.', type=str)
    parser.add_argument('--category_names ', dest="category_names", type=str, action="store")
    parser.add_argument('--top_k', dest="top_k", type=int, action="store", default=1)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    device = utils.get_device(args.gpu)
    if not os.path.exists(args.image_path):
        print(f"Image {args.image_path} doesn't exist.")
        return
    
    cat_to_name = {}
    if args.category_names:
        if not os.path.exists(f'./{args.category_names}'):
            print(f"Category names file {args.category_names} doesn't exist.")
            return
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    
    model, optimizer, _ = utils.load_model(args.checkpoint, device)
    if not model:
        return
    
    probs, classes = utils.predict(args.image_path, model, device, args.top_k)
    for p, c in zip(probs, classes):
        if args.category_names:
            print(f"Category '{cat_to_name.get(c, 'Unknown')}' with prob of {p}")
        else:
            print(f"Class {c} with prob of {p}")
    


if __name__ == "__main__":
    main()