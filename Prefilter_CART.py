import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from scipy.signal import convolve2d
import scipy.ndimage
import warnings
from skimage.feature import peak_local_max
from skimage.draw import disk
import time
start_time = time.time()
warnings.filterwarnings("ignore")
#sys.exit()
#%%
###########################################################################
#                           User Functions                                #
###########################################################################

def std_convoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)
    
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
    return np.sqrt((s2 - s**2 / ns) / ns)

#%%
###########################################################################
#                           User Variables                                #
###########################################################################

#Image padding
PAD = 200
PAD_HALF = int(PAD/2)

#Local standard deviation window size
STD_KERNEL_SIZE = 5

#Matching kernel parameters
RAD = 18
RAD_IN = 8

#Radius of extracted segment, increase for larger cells
RAD_SEG = 23

#Local maxima distance, decrease to include closer together candidates
MAX_DIS = 20

#Local maxima intensity threshold (0-1), decrease for more final candidates
THRESHOLD = 0.6 

#Input folder containing folders. Folder must contain at least 1 folder
PATH_IN = "Example_Folder"   

#Output folder, will be created if does not exist
PATH_OUT = r'Example_Output'

#%%
###########################################################################
#                     Matching Kernel and Folders                         #
###########################################################################
Path(PATH_OUT).mkdir(parents=True, exist_ok=True)    

kernel_match = np.zeros((RAD*2, RAD*2), dtype=int)
rr, cc = disk((RAD, RAD), RAD)
rr2, cc2 = disk((RAD,RAD), RAD_IN)
  
kernel_match[rr, cc] = 1
kernel_match[rr2, cc2] = -1

subfolders = [x[0] for x in os.walk(PATH_IN)]
#%%
###########################################################################
#                                 Filter                                  #
###########################################################################
for k in tqdm(range(len(subfolders))):
    dir_list = os.listdir(subfolders[k])
    filename = dir_list[0]
    img_num = 0

    if filename.count('.tif') == 1:
        name = subfolders[k]
        name_split = name.rsplit('\\',2)
        
        conc = name_split[1]
        conc = conc.zfill(4)
        trial_num = name_split[2]
        if trial_num.count('a') == 1 or trial_num.count('A') == 1:
            trial = str(1)
        elif trial_num.count('b') == 1 or trial_num.count('B') == 1:
            trial = str(2)
        elif trial_num.count('c') == 1 or trial_num.count('C') == 1:
            trial = str(3)
        trial = trial.zfill(2)  

        cart_count = 0
        
        for i in range(len(dir_list)):
    
            img_num += 1

            img_str = str(img_num)
            img_fill = img_str.zfill(3)
            name_upd = "Concentration_" + conc + "_Trial_" + trial + "_Image_" + img_fill + ".tif"
            name_upd2 = "Concentration_" + conc + "_Trial_" + trial + "_Image_" + img_fill + ".png"
            name_upd3 = "Concentration_" + conc + "_Trial_" + trial + ".txt"
            name_upd4 = "Concentration_" + conc + "_Trial_" + trial + "_Overlap_Image_" + img_fill + ".png"
            
            dir_bf = subfolders[k] + '/' + dir_list[i]
        
            image_original = cv2.imread(dir_bf, cv2.IMREAD_ANYDEPTH)
            cols, rows = np.shape(image_original)
            
            image_uint8 = np.uint8(image_original/np.max(image_original)*255)
            
            img_med = np.median(image_uint8[500:-500,500:-500])
            img_std = np.std(image_uint8[500:-500,500:-500]) / 2

            image_pad_g = np.random.normal(size=(rows+PAD,cols+PAD))
            image_pad_g = image_pad_g * img_std + img_med
            image_pad_g = image_pad_g.astype('uint8')
            
            image_pad = np.copy(image_pad_g)
            image_pad[PAD_HALF:-PAD_HALF,PAD_HALF:-PAD_HALF] = image_uint8
    
            image_local_std = std_convoluted(image_uint8,STD_KERNEL_SIZE)
            
            image_lstd_norm = image_local_std / np.max(image_local_std) *2
            image_lstd_uint8 = np.uint8(image_lstd_norm / np.max(image_lstd_norm) * 255)
            
            image_lstd_balance = image_lstd_norm - 1
            
            image_match_filt = convolve2d(image_lstd_balance, kernel_match)
            image_match_filt = image_match_filt[RAD-1:-RAD,RAD-1:-RAD]
           
            image_filt = image_match_filt
            
            min_val = np.min(image_match_filt)
            image_match_adj = image_match_filt - min_val
            
            max_val = np.max(image_match_adj)
            image_match_norm = image_match_adj / (max_val)
    
            image_thresh = np.copy(image_match_norm)
            image_thresh[image_thresh < THRESHOLD] = 0
            
            coords_peaks = peak_local_max(image_thresh, min_distance=MAX_DIS)
            coords_peaks = coords_peaks + PAD_HALF
            
            image_overlap = np.zeros((cols+PAD,rows+PAD),dtype=np.uint8)
            image_binary = np.zeros((cols+PAD,rows+PAD),dtype=int)
            image_binary_over = np.zeros((cols+PAD,rows+PAD),dtype=int)
            
            cell_count = len(coords_peaks)
            cell_count_str = str(cell_count)
            cell_count_pad  = cell_count_str.zfill(4)
            
            cart_count += cell_count
            cart_count_str = str(cart_count)
            cart_count_pad  = cart_count_str.zfill(4)
            
            overlap_stagger = 20
            overlap_count = 1
            over_row = 1
            if cell_count > 0:
                for j in range(cell_count):
                    
                    centroid = coords_peaks[j]
                    x_s = centroid[1] - RAD_SEG
                    x_e = centroid[1] + RAD_SEG
                    y_s = centroid[0] - RAD_SEG
                    y_e = centroid[0] + RAD_SEG
                    mask = np.zeros((RAD_SEG*2, RAD_SEG*2), dtype=int)
                    rr, cc = disk((RAD_SEG, RAD_SEG), RAD_SEG)
                    mask[rr, cc] = 1
                    img_cut = image_pad[y_s:y_e,x_s:x_e]
                    img_cut = img_cut * mask
                    border = 0
                    img_cut = img_cut+border
                    
                    image_binary[y_s:y_e,x_s:x_e] = image_binary[y_s:y_e,x_s:x_e] + mask                
                    
                    if overlap_count > 33:
                        over_row += 1
                        overlap_count = 1

                    x_os = (overlap_stagger + RAD_SEG*2) * overlap_count
                    x_oe = (overlap_stagger + RAD_SEG*2) * overlap_count + RAD_SEG * 2
                    y_os = (overlap_stagger + RAD_SEG*2) * over_row
                    y_oe = (overlap_stagger + RAD_SEG*2) * over_row + RAD_SEG * 2
                    
                    image_binary_over[y_os:y_oe,x_os:x_oe] = image_binary_over[y_os:y_oe,x_os:x_oe] + mask
                    image_overlap[y_os:y_oe,x_os:x_oe] = image_overlap[y_os:y_oe,x_os:x_oe] + img_cut
                    
                    overlap_count += 1
                    
            image_binary[image_binary>1.1] = 1
            image_final = image_pad * image_binary
            
            image_binary2 = np.copy(image_binary)
            image_binary2 = image_binary2 * -1
            image_binary2 = image_binary2 + 1
            
            image_border = image_binary2 * image_pad_g
            image_final = image_final + image_border
            
            image_binary3 = np.copy(image_binary_over)
            image_binary3 = image_binary3 * -1
            image_binary3 = image_binary3 + 1
            
            image_border2 = image_binary3 * image_pad_g
            image_final2 = image_overlap + image_border2
            
            image_final = image_final[PAD_HALF+1:-1*PAD_HALF,PAD_HALF+1:-1*PAD_HALF]
            filename_fin = PATH_OUT + '/' + name_upd
            cv2.imwrite(filename_fin, np.uint8(image_final))
            filename_fin2 = PATH_OUT + '/' + name_upd2
            cv2.imwrite(filename_fin2, image_uint8)
            filename_fin3 = PATH_OUT + '/' + name_upd4
            cv2.imwrite(filename_fin3, image_final2)
            
        filename_count = PATH_OUT + '/' + name_upd3
        with open(filename_count, 'w') as f:
            f.write('%d' % cart_count)
            
end_time = time.time()
print(round((end_time - start_time)/60,3))   
