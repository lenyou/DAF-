class AverageMeter():
    '''
    Computes and stores the average and current value, Copied from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    '''

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        '''reset'''
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, ncount=1):
        '''update'''
        self.val = val
        self.sum += val * ncount
        self.count += ncount
        self.avg = self.sum / self.count