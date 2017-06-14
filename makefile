HEADERS = Globalvars.h, d2func.h, d3func.h, Enfunc.h, Gradfunc.h, ConGradfunc.h, Misc.h, myhelpers.h

default: SCsimCUDA

SCsimCUDA:
	nvcc Globalvars.cu main.cu Misc.cu ConGradfunc.cu Gradfunc.cu Enfunc.cu d3func.cu d2func.cu -o SCsimCUDA -lm --relocatable-device-code true
