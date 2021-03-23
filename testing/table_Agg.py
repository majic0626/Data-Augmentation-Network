rots = 4
methods = ['MSP', 'MeanPosMax', 'MaxMax', 'MeanMax', 'JSD']
dataset = [
    'texture', 'svhn', 'places365', 'lsun',
    'cifar100', 'gaussian', 'rademacher', 'blob', 'avg'
]

performance = {d: [] for d in dataset}

metric = ['AUROC', 'AUPR', 'FPR', 'DetErr']

for m in methods:
    with open("./ckpts/noFlip/rot4/cifar10/{}.txt".format(m), 'r') as f:
        for line in f.read().split('\n')[1:-1]:
            c = line.split()
            for n in c[1:]:
                performance[c[0]].append(str(round(float(n), 1)))


all_result = []
for d in dataset:
    result = []
    for i in range(4):
        result += [performance[d][i + 4 * offset] for offset in range(5)]
    all_result.append(result)

for ix, r in enumerate(all_result):
    print(dataset[ix], ":", '&'.join(r[:len(r) // 2]))
    print(dataset[ix], ":", '&'.join(r[len(r) // 2:]))
