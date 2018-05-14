import sys

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from mpl_toolkits.mplot3d import Axes3D

from skimage import filters

import pandas as pd

def gen_points(start, num_segs, seg_len):
    def _sample_spherical(npoints, ndim=3):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec.reshape((1,3))
    start_pt = _sample_spherical(1) # returns unit vector
    start_pt *= seg_len
    pts = [start]
    pt = start_pt
    pt = pt[0]
    pt += pts[0]
    pts.append(pt)
    
    for i in range(num_segs-1):
        next_pt = get_next_pt(pts[0+i], pts[1+i], seg_len)
        pts.append(next_pt)
    return np.array(pts)


def get_next_pt(prev_pt, curr_pt, seg_length, bound=True):
    diff = curr_pt - prev_pt
#     next_pt = curr_pt + diff
    vec = [np.random.normal(diff[0], seg_length/4),
           np.random.normal(diff[1], seg_length/4),
           np.random.normal(diff[2], seg_length/4)]
    vec = np.array(vec)
    vec /= np.linalg.norm(vec)  # normalizing to magnitude of 1
    vec *= seg_length
    next_pt = curr_pt + vec
    while bound and (next_pt[0] < 0 or next_pt[0] >= 100):
        vec = [np.random.normal(diff[0], seg_length/4),
           np.random.normal(diff[1], seg_length/4),
           np.random.normal(diff[2], seg_length/4)]
        vec = np.array(vec)
        vec /= np.linalg.norm(vec)  # normalizing to magnitude of 1
        vec *= seg_length
        next_pt = curr_pt + vec
    return next_pt


# method 2
def plot_curves_on_tif(curves, tif_output_path, shape=[], reference_img_path=None, cell_bodies=False, overwrite=False, return_array=False):
    """Given a CSV file, plots the co-ordinates in the CSV on a TIF stack"""
    def _parse_int_array(arr):
        return [int(item) for item in arr]

    def _draw_circle(image, coord, size=2):
        coord = _parse_int_array(coord)
        shape_z, shape_y, shape_x = image.shape
        z_range = range(max(0, coord[0]-size), min(shape_z, coord[0]+size))
        y_range = range(max(0, coord[1]-size), min(shape_y, coord[1]+size))
        x_range = range(max(0, coord[2]-size), min(shape_x, coord[2]+size))
        
        max_dist = abs(np.linalg.norm((np.array(coord) + size) - np.array(coord)))
#         print('max_dist:', max_dist)

        mu = 256 - 32
        sigma = 32
        center_val = np.random.normal(mu, sigma)
        center_val = min(center_val, 255)
        center_val = max(center_val, 0)
#         print('center_val:', center_val)

        for z in z_range:
            for y in y_range:
                for x in x_range:
                    diff = np.array([z, y, x]) - coord
                    dist = abs(np.linalg.norm(diff))
                    if dist > size:
                        continue
#                     print('dist', dist)
                    k = 1 - float(dist) / max_dist
#                     print('k:', k)
                    val = k * center_val
#                     val = center_val
#                     print('val after:', val)
                    image[z, y, x] = val
#                     image[z, y, x] =np.random.randint(256)
#                     image[z, y, x] = 255

        return image
    
    if shape:
        shape_z, shape_y, shape_x = shape
    else:
        ref_image = tiff.imread(reference_img_path)
        if len(ref_image.shape) == 4:
            ref_image = ref_image[:,:,:,0]
        shape_z, shape_y, shape_x = ref_image.shape

    if overwrite:
        annotated_image = ref_image # pass by reference (shallow copy)
    else:
        annotated_image = np.zeros((shape_z, shape_y, shape_x))
        
#     print(np.array_equal(ref_image,annotated_image))
        
    if cell_bodies:
        cell_pt = np.random.choice(len(points))
        _draw_circle(annotated_image, points[cell_pt], size=25)
    
    for curve in curves:
        for i in range(1, len(curve)):
            diff = curve[i] - curve[i-1]
            mag = np.linalg.norm(diff)
            for j in range(0, int(mag)):
                pt = curve[i-1] + (float(j) / mag) * diff
                size = np.random.normal(1, 1.5)
                while size <= 1:
                    size = np.random.normal(1, 1.5)
                size = int(round(size)) # rounding to nearest int because range can't do floats
                annotated_image = _draw_circle(annotated_image, pt, size)
    
#     tiff.imsave('intermediate.tif', annotated_image.astype(np.uint8))
    annotated_image = filters.gaussian(annotated_image, sigma=3)
        
    tiff.imsave(tif_output_path, annotated_image.astype(np.uint8))
    if return_array:
        return annotated_image
    
    
def curves_to_swc(curves, file_name):
    count = 1
    curve_count = 1
    lines = np.array([])
    for curve in curves:
        for i, point in enumerate(curve):
            if not i:
                line = np.concatenate(([count, 3], point, [1, -1]))
            else:
                line = np.concatenate(([count, 3], point, [1, count-1]))
            if not lines.shape[0]:
                # if first element in lines
                lines = np.array([line])
                count += 1
            else:
                count += 1
#                 print(lines)
#                 print(line)
                lines = np.vstack((lines, line))
    
#     print(lines)
            
    np.savetxt(file_name, lines, fmt='%i')
    
    
def swc_to_curves(file_name, skiprows=0):
    # reading in annotations file
    ann = pd.read_csv(file_name, header=None, delim_whitespace=True, skiprows=skiprows)
    ann_points = ann[[2,3,4,6]]
    ann_points.columns = ['x', 'y', 'z', 'i']
    # print(ann_points)

    # computing curves
    ann_curves = []
    curve = []

    for index, row in ann_points.iterrows():
        if row['i'] == -1 and index != 0:
            ann_curves.append(np.array(curve))
            curve = []
        curve.append(np.array([row['x'], row['y'], row['z']]))

    ann_curves = np.array(ann_curves)
    return ann_curves
