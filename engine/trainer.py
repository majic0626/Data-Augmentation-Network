# this file contains the train loop
def Train(args, ep, trainloader, model, opt, adjust_lr, criterion, record):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for ix, sample in enumerate(trainloader):
        opt.zero_grad()
        # get data and label from loader
        data = []
        label = []
        for t in sample:  # ensure order
            data.append(sample[t]["image"])
            label.append(sample[t]["label"])
        # concat data and label
        data = torch.cat(data, 0)
        label = torch.cat(label, 0)
        data, label = data.cuda(), label.cuda()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        opt.step()
        lr = adjust_lr(
            opt=opt,
            it=(ep - 1) * len(trainloader) + ix,
            lr_init=args.lr,
            lr_end=1e-6,
            total_it=args.epoch * len(trainloader),
            warmup=0
        )
        train_loss += loss.item()
        _, predict = output.max(1)
        total += label.size(0)
        correct += predict.eq(label).sum().item()
        if (ix + 1) % 100 == 0:
            print("Learning Rate: {}".format(lr))
            print("L-train loss:{} / L-acc:{}".format(
                train_loss / (ix + 1),
                100 * correct / total))
    record["train_loss"].append(train_loss / (ix + 1))
    record["train_acc"].append(100 * correct / total)