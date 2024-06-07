from data_provider.data_loader import Dataset_Weather, Dataset_Finance, Dataset_Healthcare
from torch.utils.data import DataLoader

data_dict = {
    'weather': Dataset_Weather,
    'finance': Dataset_Finance,
    'healthcare': Dataset_Healthcare
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag in ['test', 'TEST', 'VAL', 'ALL']:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    drop_last = False
    data_set = Data(
        args,
        flag=flag,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
        #collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )
    return data_set, data_loader
