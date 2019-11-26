from ctypes import * 
l = CDLL("./do.so")
l.add(3,4)