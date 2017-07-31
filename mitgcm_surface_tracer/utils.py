import numpy as np
import os


def writebin(a, filepath, verbose=False):
    '''Reads MITgcm bin input files
    a = np.array to save
    file = filepath
    '''
    f = open(filepath, 'w')
    a.astype(dtype=np.dtype('float32').newbyteorder('>')).tofile(f)
    print('--- Written to '+filepath+' ---')


def writetxt(string, filepath, verbose=False):
    f = open(filepath, 'w')
    f.write(string)
    f.close()
    print('--- Written to '+filepath+' ---')


class writable_mds_store:
    '''adapted from @rabernat aviso processing notebooks'''
    def __init__(self, prefix, iters, suffix='data', dtype='>f4'):
        self.prefix = prefix
        self.iters = iters
        self.suffix = suffix
        self.dtype = dtype

    def __setitem__(self, idx, data):
        # first slice should be the time index
        tslice = idx[0]
        # make sure it is just one single time slice
        assert tslice.step is None
        assert (tslice.stop - tslice.start) == 1
        n = tslice.start
        fname = '%s.%010d.%s' % (self.prefix, self.iters[n], self.suffix)
        # print("Writing %s" % fname)
        data.astype(self.dtype).tofile(fname)


def readbin(file, shape):
    '''Reads MITgcm bin input files
    file = filepath to the bin
    shape = desired output shape
    '''
    f = open(file, 'r')
    a = np.fromfile(f, dtype=np.dtype('float32').newbyteorder('>'))
    a = a.reshape(shape)
    return a

# def readtxt(str, file):


def paramReadout(directory):
    # TODO write test
    '''cheap implementation to read out the data* files from mitgcm dir'''
    directory
    params = dict(check=[])
    for dfile in ['data', 'data.ptracers', 'data.diagnostics']:
        for line in open(directory+dfile):
            if '&' not in line and '#' not in line and '\n' not in line[0]:
                line_out = (line.replace('\n', '')
                            .replace(' ', '')
                            .split('=')
                            )
                if len(line_out) == 2 and len(line_out[1]) > 1:
                    while line_out[1][-1] in ['.', ',']:
                        line_out[1] = line_out[1][0:-1]
                        while line_out[1][-1] in ['.', ',']:
                            line_out[1] = line_out[1][0:-1]
                    params[dfile+'/'+line_out[0]] = line_out[1]
    return params


def dirCheck(directory, makedir):
    '''Check if directory exist and create it if not'''
    if not os.path.exists(directory) and makedir:
        os.makedirs(directory)
    out = os.path.join(directory, '')
    return out
