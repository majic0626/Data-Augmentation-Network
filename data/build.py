# build dataloader for training or testing
from data.dataset import MultiAugmentDataset
from data.augment import Rot
from torch.utils.data import DataLoader
from collections import OrderedDict


def buildLoader(args, mode):
    aug = OrderedDict()
    for i in range(args.rot):
        aug["rot_{}".format(int(i * 360 / args.rot))] = Rot(int(i * 360 / args.rot))
    print(aug)
    if mode == "train":
        # dataset for training data
        trainset = MultiAugmentDataset(
            root_dir=args.data_root + '/' + 'train',
            class_file=args.data_root + '/' + 'class.txt',
            mode='train',
            transforms=aug,
            size=args.imgSize
        )

        # loader for training data
        trainloader = DataLoader(
            trainset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.worker
        )

        return trainloader
    else:
        # dataset for testing data
        testset = MultiAugmentDataset(
            root_dir=args.data_root + '/' + 'test',
            class_file=args.data_root + '/' + 'class.txt',
            mode='test',
            transforms=aug,
            size=args.imgSize
        )

        # loader for testing data
        testloader = DataLoader(
            testset,
            batch_size=200,
            shuffle=False,
            num_workers=args.worker
        )

        return testloader
        
