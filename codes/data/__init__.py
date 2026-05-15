from torch.utils.data import DataLoader

def create_dataloaders(args):
    """create dataloader"""
    if args.dataset == 'AID':
        from data.aid import AIDataset
        # Use configured paths or fall back to defaults
        train_dir = args.data_train if args.data_train != '.' else '../datasets/AID'
        val_dir = args.data_val if args.data_val != '.' else '../datasets/AID'
        training_set = AIDataset(args, root_dir=train_dir, train=True)
        val_set = AIDataset(args, root_dir=val_dir, train=False)
    elif args.dataset == 'UCMerced':
        from data.ucmerced import UCMercedDataset
        # Use configured paths or fall back to defaults
        train_dir = args.data_train if args.data_train != '.' else '/root/autodl-tmp/TransENet_base/datasets/TransENet/UCMerced'
        val_dir = args.data_val if args.data_val != '.' else '/root/autodl-tmp/TransENet_base/datasets/TransENet/UCMerced'
        training_set = UCMercedDataset(args, root_dir=train_dir, train=True)
        val_set = UCMercedDataset(args, root_dir=val_dir, train=False)
    elif args.dataset == 'DIV2K':
        from data.div2k import DIV2KDataset
        # Use configured paths or fall back to defaults
        train_dir = args.data_train if args.data_train != '.' else '../datasets/DIV2K'
        val_dir = args.data_val if args.data_val != '.' else '../datasets/DIV2K'
        training_set = DIV2KDataset(args, root_dir=train_dir, train=True)
        val_set = DIV2KDataset(args, root_dir=val_dir, train=False)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s ' % args.dataset)

    loader_kwargs = {
        'num_workers': args.n_threads,
        'pin_memory': not args.cpu
    }
    if args.n_threads > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    dataloaders = {
        'train': DataLoader(
            training_set,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kwargs
        ),
        'val': DataLoader(
            val_set,
            batch_size=args.val_batch_size,
            shuffle=False,
            **loader_kwargs
        )
    }

    return dataloaders
