rots = [1, 2, 3, 4, 5, 6]
method = 'JSD'
dataset = [
    'texture', 'svhn', 'places365', 'lsun',
    'cifar100', 'gaussian', 'rademacher', 'blob', 'avg'
]

performance = {d: [] for d in dataset}

metric = ['AUROC', 'AUPR', 'FPR', 'DetErr']

for r in rots:
    with open("./ckpts/noFlip/rot{}/cifar10/{}.txt".format(r, method), 'r') as f:
        for line in f.read().split('\n')[1:-1]:
            c = line.split()
            for n in c[1:]:
                performance[c[0]].append(str(round(float(n), 1)))

all_result = []
for d in dataset:
    result = []
    for i in range(4):
        result += [performance[d][i + 4 * offset] for offset in range(len(rots))]
    all_result.append(result)
    # print(d, ":", '&'.join(result))

for ix, r in enumerate(all_result):
    print(dataset[ix], '&'.join(r[:len(r) // 2]))
    print(dataset[ix], '&'.join(r[len(r) // 2:]))
