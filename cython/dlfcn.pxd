from libc.stdio cimport printf

## ----- Library loader ----- ##
cdef extern from *:
    ctypedef char const_char "const char"

cdef extern from 'dlfcn.h' nogil:
    void* dlopen(const_char *filename, int flag)
    char *dlerror()
    void *dlsym(void *handle, const_char *symbol)
    int dlclose(void *handle)
    unsigned int RTLD_NOW
    unsigned int RTLD_GLOBAL
## -------------------------- ##


## -- load func_name from library in libpath -- ##
cdef inline void* load_func(char* libpath, char* func_name):

    cdef:
        void *handle = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL)
    
    if handle == NULL:
        printf("%s\n", dlerror())
        raise ImportError
    
    cdef void* func_ = dlsym(handle, func_name)
    if func_ == NULL:
        printf("%s\n", dlerror())
        raise ImportError
    return func_
## -------------------------------------------- ##


