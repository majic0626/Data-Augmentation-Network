import torch
from utils.tools import record_saver

# this file contains the test loop

def Test(args, ep, testloader, model, criterion, record, best_acc):
    model.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for ix, sample in enumerate(testloader):
            data = []
            label = []
            for t in sample:  # ensure the order
                data.append(sample[t]["image"])
                label.append(sample[t]["label"])
            data = torch.cat(data, 0)
            label = torch.cat(label, 0)
            data = data.cuda()
            label = label.cuda()
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            _, predict = output.max(1)
            total += label.size(0)
            correct += predict.eq(label).sum().item()
        test_acc = 100 * correct / total
        print("L-test loss:{} / L-acc:{}".format(test_loss / (ix + 1), test_acc))
        record["test_loss"].append(test_loss / (ix + 1))
        record["test_acc"].append(100 * correct / total)
        if test_acc > best_acc:
            print("save the new best model: {} || old best model: {}".format(test_acc, best_acc))
            best_acc = test_acc
            torch.save({"cnn": model.state_dict(), "epoch": ep}, args.dir_save_ckpt + '/' + "best.pt")
        print("save the last model: {} || best model: {}".format(test_acc, best_acc))
        torch.save({"cnn": model.state_dict(), "epoch": ep}, args.dir_save_ckpt + '/' + "last.pt")
        record_saver(args, record)