from dataloaders.datasets import combine_dbs, pascal, sbd, cocostuff
from torch.utils.data import DataLoader
from mypath import Path

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cocostuff10k':
        train_set = cocostuff.CocoStuff10k(split='train',
                                                root = Path.db_root_dir('cocostuff10k'),
                                                ignore_label=255,
                                                mean_bgr=(104.008, 116.669, 122.675),
                                                augment=True,
                                                base_size=args.base_size,
                                                crop_size=args.crop_size,
                                                scales=(0.5, 0.75, 1.0, 1.25, 1.5),
                                                flip=True)

        val_set = cocostuff.CocoStuff10k(split='test',
                                                root = Path.db_root_dir('cocostuff10k'),
                                                ignore_label=255,
                                                mean_bgr=(104.008, 116.669, 122.675),
                                                augment=False,
                                                base_size=args.base_size,
                                                crop_size=args.crop_size,
                                                scales=(0.5, 0.75, 1.0, 1.25, 1.5),
                                                flip=False)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        # val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        test_loader = None
        return train_loader,val_loader,test_loader,num_class
    else:
        raise NotImplementedError

