from tools.train import get_args_parser, main

if __name__=="__main__":
    args = get_args_parser().parse_args()
    args.batch_size = 64
    print(args)
    # main(args)