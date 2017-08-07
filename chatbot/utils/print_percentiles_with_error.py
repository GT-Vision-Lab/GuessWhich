import argparse
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-json_input', type=str)
    args = parser.parse_args()

    data = json.load(open(args.json_input, 'r'))
    ranks = data['gt_ranks']

    rounds = {i:[] for i in range(10)}
    for i in range(len(ranks)):
        rounds[i%10].append(ranks[i])

    total = 9628
    means, mins, maxs = [], [], []
    means_p, mins_p, maxs_p = [], [], []
    print 'Computing percentiles...'
    for i in range(10):
        arr = np.array(rounds[i])
        means.append(arr.mean())
        se = arr.std() / np.sqrt(total)
        mins.append(arr.mean() - se)
        maxs.append(arr.mean() + se)

        means_p.append(100 * (9628.0 - arr.mean()) / 9628.0)
        mins_p.append(100 * (9628.0 - arr.mean() - se) / 9628.0)
        maxs_p.append(100 * (9628.0 - arr.mean() + se) / 9628.0)

    print means
    print mins
    print maxs
    print means_p
    print mins_p
    print maxs_p
