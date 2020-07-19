import torch
import os

def load_model(net, load_dir, load_weights=True, force_parallel=True):
    '''
    loading model function
    :param net: defined network
    :param load_dir: the path for loading
    :return: network
    '''
    if load_weights:
        print('load pretrain model: ', load_dir)
        checkpoint = torch.load(load_dir)
        pretrained_dict = checkpoint['state_dict']
        model_dict = net.state_dict()
        for k,v in model_dict.items():
            #print(v)
            if k in pretrained_dict.keys():
                if not v.shape == pretrained_dict[k].shape:
                    if len(v.shape)==len(pretrained_dict[k].shape):
                        if len(v.shape)==1:
                            model_dict[k] = pretrained_dict[k][:v.shape[0]]
                        elif len(v.shape)==2:
                            model_dict[k] = pretrained_dict[k][:v.shape[0], :v.shape[1]]
                        elif len(v.shape)==3:
                            model_dict[k] = pretrained_dict[k][:v.shape[0], :v.shape[1], :v.shape[2]]
                        elif len(v.shape)==4:
                            model_dict[k] = pretrained_dict[k][:v.shape[0], :v.shape[1], :v.shape[2], :v.shape[3]]
                        elif len(v.shape)==5:
                            model_dict[k] = pretrained_dict[k][:v.shape[0], :v.shape[1], :v.shape[2], :v.shape[3], :v.shape[4]]
                else:
                    model_dict[k] = pretrained_dict[k]
        # net.load_state_dict(checkpoint['state_dict'])
        net.load_state_dict(model_dict)
    return net


def save_model(net, epoch, save_dir, prefix = 'model_'):
    '''
    saving model function
    :param net:  defined network
    :param epoch: current saved epoch
    :param save_dir: the path for saving
    :return: None
    '''

    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'save_dir': save_dir,
        'state_dict': state_dict},
        os.path.join(save_dir, prefix + 'at_epoch_%03d.dat' % (epoch + 1)))


