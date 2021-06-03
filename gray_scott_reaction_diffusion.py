#!/usr/bin/env python3

import os, sys, random
import argparse
import numpy as np
from scipy.ndimage.filters import laplace, gaussian_filter
from scipy import signal

from PIL import Image

"""
gray-scott diff.eqs.:
===================================================
A' = A + (D_A grad^2(A) - A*B^2 + f(1 - A))*delta_t 
B' = B + (D_B grad^2(B) + A*B^2 - (k + f)B)*delta_t
====================================================
A,B = chemicals
delta_t = timestep
D_A,D_B = Diffusion rates
--> see here: http://karlsims.com/rd.html
"""

parser = argparse.ArgumentParser()
# parser.add_argument("rgb_image", help="set path to u_image")
parser.add_argument("-padding", default=16, required=False, type=int)
parser.add_argument("-size", default=256, required=False, type=int)
parser.add_argument("-m", "--method", default=1, required=False, type=int)
parser.add_argument("-u", "--image_u", help="set path to u_image")
parser.add_argument("-v", "--image_v", help="set path to v_image")
# parser.add_argument("-c", "--cropsize", default=[128, 128], type=int, nargs='+', help="Set dimensions of crop", required=True)
args = parser.parse_args()

save_folder = "./output_folder"
save_interval = 128 # save every 50th iteration into image
if not os.path.exists(save_folder):
	os.mkdir(save_folder)

# img_u = Image.open(args.image_u)
# img_v = Image.open(args.image_v)
# size = img.size 

print("size is {} x {}".format(args.size, args.size))
dimensions = [args.size, args.size]

initial_u = np.array(Image.open(args.image_u).resize((dimensions[1],dimensions[0]),1).convert("L"), dtype=np.float32)/255
initial_v = np.array(Image.open(args.image_v).resize((dimensions[1],dimensions[0]),1).convert("L"), dtype=np.float32)/255

parameters = {
"Bacteria_1":   [0.16, 0.08, 0.035, 0.065],
"Bacteria_2":   [0.14, 0.06, 0.035, 0.065],
"Coral":        [0.16, 0.08, 0.060, 0.062],
"Fingerprint":  [0.19, 0.05, 0.060, 0.062],
"Spirals":      [0.10, 0.10, 0.018, 0.050],
"Spirals_Dense":[0.12, 0.08, 0.020, 0.050],
"Spirals_Fast": [0.10, 0.16, 0.020, 0.050],
"Unstable":     [0.16, 0.08, 0.020, 0.055],
"Worms_1":      [0.16, 0.08, 0.050, 0.065],
"Worms_2":      [0.16, 0.08, 0.054, 0.063],
"Zebrafish":    [0.16, 0.08, 0.035, 0.060],
"myrule":       [0.16, 0.08, 0.09 , 0.066],
}

Du, Dv, F, k = parameters["myrule"]

F = F * np.ones(dimensions, dtype=np.float32)
k = k * np.ones(dimensions, dtype=np.float32)

a = parameters["Bacteria_1"]
b = parameters["Coral"]

# method = 1
if args.method == 1:
	Du = np.zeros(dimensions, dtype=np.float32)
	Dv = np.zeros(dimensions, dtype=np.float32)
	for i in range(dimensions[0]):
		for j in range(dimensions[1]):
			r = np.sqrt((i - 1.5*dimensions[0]/2)**2 + (j - 1.5*dimensions[1]/2)**2) * 2 / (dimensions[0] + dimensions[1])
			val = 0.5 + 0.5 * np.cos(-80*(r - 0.25)**2)
			Du[i,j] = (a[0]*val + b[0]*(1-val))
			Dv[i,j] = (a[1]*val + b[1]*(1-val))
			F[i,j] = (a[2]*val + b[2]*(1-val))
			k[i,j] = (a[3]*val + b[3]*(1-val))

print("padding = {}".format(args.padding))

if args.method == 2:
	for i in range(args.padding, dimensions[0] - args.padding):
		for j in range(args.padding, dimensions[1] - args.padding):
			F[i,j] = 0.01 + (0.080 - 0.01) * i / dimensions[0]
			k[i,j] = 0.040 + (0.080 - 0.040) * j / dimensions[1]

F = gaussian_filter(F, 4)
k = gaussian_filter(k, 4)

# Image.fromarray(np.uint8(k*255)).save("k_image.png")

pre_run = 0
iters = 50000

u = initial_u
v = initial_v

def save_state(_folder, _u, _v, _index):
	u_pic = Image.fromarray(np.uint8(_u*255))
	u_pic.save("{}/u_{:0>3}.png".format(_folder,_index))
	v_pic = Image.fromarray(np.uint8(_v*255))
	v_pic.save("{}/v_{:0>3}.png".format(_folder,_index))

# main loop that saves images
for i in range(iters):
	# calculate values of the laplace-operator for u and v at each pixel
	lapu = laplace(u, mode="wrap")
	lapv = laplace(v, mode="wrap")

	temp_u = u
	uvv = u*v*v
	u += (Du*lapu - uvv +  F *(1-temp_u))
	v += (Dv*lapv + uvv - (F+k)*v)

	if np.mod(i, save_interval) == 0:
		print("iteration {:0>3}, saving image {:0>3}".format(i,i // save_interval))
		save_state(save_folder, u, v, i//save_interval)
