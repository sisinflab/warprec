
dataset_path = 'data/lastfm-1k/userid-timestamp-artid-artname-traid-traname.tsv'
dataset_path_fix = 'data/lastfm-1k/userid-timestamp-artid-artname-traid-traname_fix.tsv'

elems = []
print(f"Processing dataset file: {dataset_path}")
with open(dataset_path, 'r') as f:
    with open(dataset_path_fix, 'w') as fw:
        for line in f.readlines():
            splits = line.strip().split('\t')
            elems.append(len(splits))
            new_line = [splits[0], splits[2], splits[1]]
            fw.write('\t'.join(new_line) + '\n')
