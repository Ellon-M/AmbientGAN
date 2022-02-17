import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd 
cimport numpy as cnp 

import pickle

cnp.import_array()

def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass



cpdef unpickleBytes():
    cdef list result = []
    with open('message_bytes.pkl','rb') as f:
        for byte in pickleLoader(f):
           result.append(byte)
    
    return result


cpdef unpickleBytesAsArray():
    cdef list result = []
    with open('message_bytes.pkl','rb') as f:
        for byte in pickleLoader(f):
           result.append(byte)

    cdef cnp.ndarray[unsigned char, ndim=2] nd_bytes_arr = np.asarray(result)
    return nd_bytes_arr

