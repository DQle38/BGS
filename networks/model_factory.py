import torch


class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, taskcla):
        for_icarl = True if 'icarl' in args.trainer else False
        if args.dataset == 'split_cifar_100s':
            import networks.resnet_cifar as res
            network = res.resnet56(taskcla=taskcla, continual=args.continual,
                                   for_icarl=for_icarl)

        elif args.dataset in ['celeba_two', 'celeba', 'split_imagenet_100c']:
            import networks.resnet as conv
            network = conv.resnet18(taskcla=taskcla, continual=args.continual, pretrained=args.pretrained,
                                    for_icarl=for_icarl)

        else:
            raise NotImplementedError

        ########################################################################################################################
        if args.modelpath is not None:
            dic = torch.load(args.modelpath)
            if 'fc.weight' in dic.keys() and 'fc.0.weight' in network.state_dict().keys():
                temp_dict = network.state_dict()
                for key in dic.keys():
                    if 'fc' in key:
                        _key = 'fc.0.weight' if 'weight' in key else 'fc.0.bias'
                        temp_dict[_key] = dic[key]
                    else:
                        temp_dict[key] = dic[key]
                network.load_state_dict(temp_dict)
            else:
                network.load_state_dict(dic)
        ########################################################################################################################
        return network
