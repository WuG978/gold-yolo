from tools.train import get_args_parser, main

if __name__=="__main__":
    args = get_args_parser().parse_args()
    main(args)