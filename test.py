# from tools.train import get_args_parser, main
from tools.eval import get_args_parser, main
# from tools.infer import get_args_parser, main

if __name__ == "__main__":
    # args = get_args_parser().parse_args()
    args = get_args_parser()
    main(args)

