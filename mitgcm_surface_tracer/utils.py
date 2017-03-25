import numpy as np
import scipy.interpolate as spi
import os

def writebin(a,file):
    '''Reads MITgcm bin input files
    a = np.array to save
    file = filepath
    '''
    f = open(file, 'w')
    a.astype(dtype=np.dtype('float32').newbyteorder ('>')).tofile(f)

def readbin(file,shape):
    '''Reads MITgcm bin input files
    file = filepath to the bin
    shape = desired output shape
    '''
    f = open(file, 'r')
    a = np.fromfile(f, dtype=np.dtype('float32').newbyteorder ('>'))
    a = a.reshape(shape)
    return a

#TODO write test
def paramReadout(directory):
    '''cheap implementation to read out the data* files from mitgcm dir'''
    directory
    params = dict(check=[])
    for dfile in ['data','data.ptracers','data.diagnostics']:
        for line in open(directory+dfile):
            if '&' not in line and '#' not in line and '\n' not in line[0]:
                line_out = (line.replace('\n','')
                                .replace(' ','')
                                .split('=')
                                )
                if len(line_out)==2 and len(line_out[1])>1:
                    while line_out[1][-1] in ['.',',']:
                        line_out[1] = line_out[1][0:-1]
                        while line_out[1][-1] in ['.',',']:
                            line_out[1] = line_out[1][0:-1]
                    params[dfile+'/'+line_out[0]] = line_out[1]
    return params

def dirCheck(directory,makedir):
    '''Check if directory exist and create it if not'''
    if not os.path.exists(directory) and makedir:
        os.makedirs(directory)
    out = os.path.join(directory, '')
    return out
