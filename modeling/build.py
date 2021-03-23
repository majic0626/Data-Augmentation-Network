from modeling.models import WideResNet

# build wide resnet according to args
def buildModel(args):
    if args.pretrain == 'x':
        model = WideResNet(depth=40,
                           num_classes=args.classNum,
                           widen_factor=2,
                           dropRate=0.3,
                           final_pool=args.imgSize // 4)
    else:
        # ckpt from imagenet-based pretrained model
        model = WideResNet(depth=40,
                           num_classes=args.classNum,
                           widen_factor=2,
                           dropRate=0.0,
                           final_pool=args.imgSize // 4)
        _ckpt = torch.load(args.pretrain)
        ckpt = OrderedDict()
        for k in _ckpt.keys():
            if k.split(".")[0] == "module":
                ckpt[k[7:]] = _ckpt[k]
            else:
                ckpt[k] = _ckpt[k]
        model.load_state_dict(ckpt)
        model.fc = nn.Linear(128, args.classNum)
    
    return model
