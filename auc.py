import random, sys

def read_data(path):
    scores, targets = [], []
    for line in open(path, "rt"):
        rows = line.split("\t")
        if len(rows) < 4 or rows[0] == "Instance":
            continue
        scores.append(float(rows[3]))
        targets.append(int(rows[2]))
        #if len(scores) > 1000:
        #    break
    return scores, targets

def auc(scores, targets):
    n = len(scores)
    idx = range(n)
    idx = sorted(idx, key = lambda x: scores[x])
    neg, pos = 1e-38, 1e-38
    tp, fp = 0, 0
    for i in idx:
        if targets[i] == 0:
            neg += 1
            fp += 1
        else:
            pos += 1
            tp += 1
    tpr, fpr = [1.0], [1.0]
    last = scores[idx[0]]
    for i in idx:
        if scores[i] > last + 1e-10:
            tpr.append(tp / pos)
            fpr.append(fp / neg)
            last = scores[i]
        if targets[i] == 0:
            fp -= 1
        else:
            tp -= 1
    tpr.append(tp / pos)
    fpr.append(fp / neg)
    auc = 0.0
    for i in xrange(len(tpr) - 2, -1, -1):
        auc += ((fpr[i] - fpr[i + 1]) * (tpr[i + 1] + tpr[i]))
    return auc / 2.0

def main():
    scores, targets = read_data(sys.argv[1])
    aucs = []
    for i in xrange(100):
        idx = [random.randint(0, len(scores) - 1) for t in xrange(len(scores))]
        aucs.append(auc([scores[t] for t in idx], [targets[t] for t in idx]))
        print aucs[-1]
    print "Average:", sum(aucs) / len(aucs)


if __name__ == "__main__":
    main()
