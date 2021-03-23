from torch import optim

def getOptim(args, model):
    # get optimizer to train the model
    opt = optim.SGD(
        list(model.parameters()),
        lr=args.lr,
        momentum=args.momentnum,
        weight_decay=args.decay,
        nesterov=True
    )
    return opt