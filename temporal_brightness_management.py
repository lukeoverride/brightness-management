# Copyright (C) 2025 Luca Surace - USI Lugano
# License CC BY-NC-SA

import cv2
import numpy as np
from colour.contrast import retinal_illuminance_Barten1999, optical_MTF_Barten1999, sigma_Barten1999, contrast_sensitivity_function_Barten1999
import matplotlib.pyplot as plt
import glob, os
import time
from scipy.ndimage import convolve
from scipy.optimize import minimize
from scipy.optimize import Bounds, NonlinearConstraint
import itertools
from scipy.interpolate import interpn, CubicSpline, make_interp_spline
from argparse import ArgumentParser
import colour


def generate_laplacian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)

    laplacian_pyramid = [gaussian_pyramid[levels - 1]]
    currPixel = (100,100)
    for i in range(levels - 1, 0, -1):
        expanded = cv2.pyrUp(gaussian_pyramid[i])
        expanded = adapt_dimension(gaussian_pyramid[i - 1], expanded)
        laplacian = gaussian_pyramid[i - 1] - expanded
        laplacian_pyramid.append(laplacian)

    gaussian_pyramid.reverse()
    return gaussian_pyramid, laplacian_pyramid


def adapt_dimension(img1, img2):
    while (img1.shape != img2.shape):
        if (img1.shape[0] != img2.shape[0]):
            img2 = img2[1:,:]
        else:
            img2 = img2[:,1:]
    return img2


def get_avg_luminance(currentPyr, pyrDown):
    pyrDown = cv2.resize(pyrDown,(currentPyr.shape[1],currentPyr.shape[0]),interpolation=cv2.INTER_CUBIC)
    allAverages = pyrDown + 1e-9
    return allAverages


def compute_luminance_contrast(gaussian_pyramid, laplacian_pyramid):
    max_spatial_freq = 36.14
    avg_spatial_freq = max_spatial_freq / (2 ** len(laplacian_pyramid))
    pooled_value = 0.0

    for i in range(2, len(laplacian_pyramid)):
        avg_spatial_freq *= 2

        
        all_avg_luminance = get_avg_luminance(gaussian_pyramid[i],gaussian_pyramid[i-2])

        contrast = contrast_sensitivity_function_Barten1999(avg_spatial_freq, E=retinal_illuminance_Barten1999(np.mean(all_avg_luminance),3))
        laplacian_pyramid[i] = np.abs( laplacian_pyramid[i] / (all_avg_luminance + 1e-9) * contrast) #.../ (all_avg_luminance + 1e-9)) * contrast )

        alpha = 0.7; beta = 0.2
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
        avg_vicinato = convolve(laplacian_pyramid[i],kernel,mode='constant',cval=0) / np.count_nonzero(kernel)
        perceived_luminance_contrast = (np.sign(laplacian_pyramid[i]) * (laplacian_pyramid[i] ** alpha) )/ ( 1 + avg_vicinato ** beta )

        laplacian_pyramid[i] = np.where(perceived_luminance_contrast > 1, 1, 0)
        pooled_value += np.mean(laplacian_pyramid[i])

    if (strategy == 'xor'):
        return laplacian_pyramid
    else:
        return pooled_value/len(laplacian_pyramid)



def avg_with_xor(visible_contrast_orig, visible_contrast_deemed):
    pooled_value = 0.0

    for i in range(2, len(visible_contrast_orig)):
        area_of_lost_contrast = np.logical_xor(visible_contrast_orig[i], visible_contrast_deemed[i])
        pooled_value += np.mean(area_of_lost_contrast)

    return pooled_value/len(visible_contrast_orig)



def reconstruct_image(laplacian_pyramid):
    image = laplacian_pyramid[0]
    for i in range(1, len(laplacian_pyramid)):
        expanded = cv2.pyrUp(image)
        image = expanded + laplacian_pyramid[i]
    return image


def convert_to_brightness_map(image):
    image = np.asarray(image, dtype=np.float32)
    image /= 255.0
    if (len(image.shape) == 2):
        brightness_map = image
    else:
        brightness_map = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]

    return brightness_map


def luminance_fitting(brightness_map, multiplier):  
    lum = np.power(brightness_map,1.95) * (max_nits_varjo - 0.2) + 0.2
    lum *= multiplier
    
    return np.clip(lum,0.2,max_nits_varjo)
    

def compute_visible_contrast(image_path, n_lapl_levels, brightness_multiplier):
    original_image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    brightness_map = convert_to_brightness_map(original_image)
    luminance_map = luminance_fitting(brightness_map, brightness_multiplier)
    gaussian_pyramid, laplacian_pyramid = generate_laplacian_pyramid(luminance_map, n_lapl_levels)
    lum_contr = compute_luminance_contrast(gaussian_pyramid, laplacian_pyramid)

    return lum_contr



def compute_luminance_avg():
    luminance_avg = np.empty(len(tempo))
    n = 0
    for hdr_frame_n in tempo:
        original_image = cv2.imread(frames[hdr_frame_n], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        brightness_map = convert_to_brightness_map(original_image)
        luminance_map = luminance_fitting(brightness_map, 1.0)
        luminance_avg[n] = np.mean(luminance_map)
        n += 1
    return luminance_avg



def compute_contrast_difference(write_file):
    contrast_difference = np.zeros((bright_mults.shape[0],len(tempo)))
    orig_visible_contrast_array = np.array([])

    if (strategy != 'xor'):
        for hdr_frame_n in tempo:
            orig_visible_contrast_array = np.append(orig_visible_contrast_array,compute_visible_contrast(frames[hdr_frame_n], n_lapl_levels, 1.0))

    for mult in range(bright_mults.shape[0]):
        sub_frame_n = 0
        for hdr_frame_n in tempo:
            if (strategy == 'xor'):
                lum_contr_orig = compute_visible_contrast(frames[hdr_frame_n], n_lapl_levels, 1.0)
                lum_contr_deem = compute_visible_contrast(frames[hdr_frame_n], n_lapl_levels, bright_mults[mult])
                contrast_difference[mult, sub_frame_n] = np.abs(avg_with_xor(lum_contr_orig, lum_contr_deem))
            else:
                deemed_visible_contrast = compute_visible_contrast(frames[hdr_frame_n], n_lapl_levels, bright_mults[mult])
                contrast_difference[mult, sub_frame_n] = np.abs(1 - deemed_visible_contrast / orig_visible_contrast_array[sub_frame_n])

            print(mult, sub_frame_n, contrast_difference[mult, sub_frame_n])
            sub_frame_n += 1
    contrast_difference = np.clip(contrast_difference,0.0,1.0)
    np.savetxt(write_file,contrast_difference)
    return contrast_difference


def get_slopes(luminance_current):
    slopes_nits_per_sec = np.array([13.8,13.3,9.6,9.2,   3.8,2.7,2.4,1.3]) / 5
    slopes_nits_per_frame = slopes_nits_per_sec * 2

    luminances_measured = np.array([104.4, 83.9, 63.6, 43.5, 23.4, 18.3, 14.1, 8.8])

    z = np.polyfit(luminances_measured,np.log(slopes_nits_per_frame),4)
    p = np.poly1d(z)
    myslope = np.exp(p(luminance_current))

    return myslope


def separate_video_frames(chosen_video):
    vidcap = cv2.VideoCapture(args.input_video_path)
    success,image = vidcap.read()
    count = 0
    print('Writing video frames...')
    while success:
      cv2.imwrite("./{}_frames/{}.jpg".format(chosen_video,str(count).zfill(4)), image )   
      success,image = vidcap.read()
      count += 1
    print('Done.')


def get_power(luminance):
    i = np.array([0.35,0.28,0.22,0.16,0.10,0.08,0.07,0.06])
    powers = 5 * i
    lum_pi = np.array([750,620,440,290,128,48,8.4,0.02])
    z = np.polyfit(lum_pi,powers,1) 
    fn = np.poly1d(z)

    return fn(luminance)


def contrast_loss_difference(brightnesses):
    points = (bright_mults,tempo)
    contrasti = interpn(points, contrast_difference, (brightnesses,tempo), bounds_error=False, fill_value=target_avg_brightness) 
    diff = np.sum(  np.abs( np.diff(contrasti[combinations]) ) , 0 )  
    return diff[0]

def br_avg_con(b):
    luminance_current = max_nits_rapsberry * b
    luminance_current = np.clip(luminance_current,0.02,max_nits_rapsberry)
    powers = np.apply_along_axis(get_power,0,b)
    return np.mean(b) - target_avg_brightness


def br_slope_con(b):
    slope = np.diff(b)
    return np.sum( np.abs( slope - get_slopes(b[:-1]) ) )



def plot_ours_vs_baseline(success):
    fig, axs = plt.subplots(2,1,sharex='col')
    axs[0].set_title('{}, strategy : {}, \n mean : {}, slope : {}'.format(chosen_video, strategy,
        str(np.round(np.mean(optimized_brightness),2)), str(slope_constraint) ))
    axs[0].plot(allframes, interp_brightness, label='optimized',color='#088163',alpha=0.8)
    axs[0].set_ylabel('brightness factor')
    axs[0].set_ylim(-0.1,1.1)

    axs[0].plot(allframes,baseline_brightness,label='baseline',color='#b4940d',alpha=0.8)
    axs[0].legend()

    contrasti_ottimi = np.array([])
    contrasti_baseline = np.array([])
    for i in subframes:
        contrasti_ottimi = np.append(contrasti_ottimi, np.interp(optimized_brightness[i], bright_mults, contrast_difference[:,i]) )
        contrasti_baseline = np.append(contrasti_baseline, np.interp(target_avg_brightness, bright_mults, contrast_difference[:,i]) )

    axs[1].set_title('Loss of visible contrast')
    axs[1].set_xlabel('time (frames)')
    axs[1].set_ylabel('contrast loss')
    axs[1].plot(allframes,CubicSpline(tempo,contrasti_ottimi,
                                        bc_type='natural')(allframes),label='optimized',color='#088163',linestyle='dashed',alpha=0.8)
    axs[1].plot(allframes,CubicSpline(tempo,contrasti_baseline,
                                        bc_type='natural')(allframes),label='baseline',color='#b4940d',linestyle='dashed',alpha=0.8)
    axs[1].legend() 


    plt.show()





parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="input_video_path",
                    help="the input video file")
parser.add_argument("-sc", "--slope_constraint", dest="slope_constraint", action="store_true", default=False,
                    help="the maximum slope of the contrast loss between subsequent frames")
parser.add_argument("-tb", "--target_brightness", dest="target_avg_brightness", default=0.6, type=np.float32,
                    help="the target average brightness for energy savings")
parser.add_argument("-c", "--strategy", dest="strategy", default="standard", 
                    help="the strategy for computation of the contrast loss", metavar="[standard, xor]")
parser.add_argument("-l", "--loss", dest="loss", action="store_true", help="compute the contrast loss for different bright multipliers, otherwise it loads data from file")
parser.add_argument("-p", "--plot", dest="show_plot", action="store_true", help="shows the brightness - contrast loss plots")

args = parser.parse_args()


chosen_video = (os.path.normpath(args.input_video_path).split('\\')[-1])[:-4]

if (not os.path.exists('./{}_frames'.format(chosen_video))):
    os.mkdir('./{}_frames'.format(chosen_video))
    separate_video_frames(chosen_video)
if (not os.path.exists('./logs')):
    os.mkdir('./logs')

n_lapl_levels = 12
strategy = args.strategy
bright_mults = np.linspace(0.2,1.0,5)
max_nits_varjo = 104.4
max_nits_rapsberry = 750
frame_path = './{}_frames/*.jpg'.format(chosen_video)

frames = glob.glob(frame_path)
tempo = np.arange(0,len(frames),10)
subframes = np.arange(0,len(tempo))
allframes = np.arange(0, len(frames))


if (args.loss):
    luminance_medi = compute_luminance_avg()
    np.savetxt('./logs/luminance_mean_{}_samples={}.txt'.format(chosen_video,len(tempo)),luminance_medi)
    contrast_difference = compute_contrast_difference('./logs/contrast_difference_{}_{}_samples={}.txt'.format(strategy,chosen_video,len(tempo)))
else:
    luminance_medi = np.loadtxt('./logs/luminance_mean_{}_samples={}.txt'.format(chosen_video,len(tempo)),dtype=np.float32)
    contrast_difference = np.loadtxt('./logs/contrast_difference_{}_{}_samples={}.txt'.format(strategy,chosen_video,len(tempo)),dtype=np.float32)


target_avg_brightness = args.target_avg_brightness
baseline_avg_brightness = target_avg_brightness
slope_constraint = args.slope_constraint
baseline_avg_luminance = max_nits_rapsberry * target_avg_brightness
baseline_power = np.round(get_power(baseline_avg_luminance),3)
print("Simulations of power consumption on Rapsberri PI Display 7\" \nPower of baseline sequence: {} W".format(baseline_power))

init_brightnesses = np.full(len(tempo), target_avg_brightness)
combinations = np.array(list(itertools.combinations(subframes,2)))


cons = [{'type': 'eq', 'fun': br_avg_con} ]
meth = "SLSQP"
opzioni = {'maxiter': 1000, 'ftol': 1e-20}
success = 'n/a'

if (slope_constraint):
    cons.append({'type': 'eq', 'fun': br_slope_con})

optimization = minimize(contrast_loss_difference, init_brightnesses, bounds=Bounds(0,1), constraints=cons, method=meth, options=opzioni)
optimized_brightness = optimization.x
success = optimization.success

baseline_brightness = np.full(len(frames),baseline_avg_brightness)


interp_brightness = CubicSpline(tempo,optimized_brightness,bc_type='natural')(allframes)

optimized_luminance_varjo = luminance_medi * optimized_brightness
optimized_luminance_varjo = np.clip(optimized_luminance_varjo,0.2,max_nits_varjo)
myspline = make_interp_spline(tempo,optimized_luminance_varjo, k=2)
slope = myspline.derivative()(tempo)

optimized_luminance = max_nits_rapsberry * optimized_brightness
optimized_power = np.apply_along_axis(get_power,0,optimized_luminance)

powers_max = np.round(get_power(max_nits_rapsberry),3)
print("Power of optimized sequence (avg): {} W\nPower of original full brightness sequence (avg): {} W".format(np.round(np.mean(optimized_power),3),powers_max))

if (args.show_plot):
    plot_ours_vs_baseline(success)
if (args.slope_constraint):
    print("Slope violated for {} over {} frames".format(np.count_nonzero(np.abs(slope) > get_slopes(optimized_luminance_varjo)), slope.shape[0]) )

timestamps = allframes/30

np.savetxt('./ours_{}_{}.txt'.format(chosen_video,str(np.round(np.mean(optimized_brightness),2))),np.stack((timestamps,interp_brightness), axis=-1))
print("Average brightness factor of optimized sequence: {}".format(np.round(np.mean(optimized_brightness),2)))

