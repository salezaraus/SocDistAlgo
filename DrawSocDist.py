# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:58:27 2021

@author: salez
"""

import pandas as pd 
import os 
import numpy as np 
import cv2
from itertools import combinations  
import multiprocessing as mp

# --------------------
# USER CONFIGURATIONS
# --------------------
# Please adjust these paths according to your setup before running the script.
PKL_FILE = 'output1.pkl'  # Path to the .pkl file with bounding box information.
OUTPUT_DIR = 'Annotated/3'  # Directory where annotated images will be stored.
IMAGE_ROOT_DIR = '../../Corrected Batch 3'  # Directory of images to be analyzed.

# Default Parameters - Adjust if necessary.
FL = 1290  # Focal length in pixels. Adjust based on camera calibration.
W_ACTUAL = 2.1  # Assumed actual width of a pedestrian in feet.

# Column indices for bounding box data.
imgname_idx, xmin_idx, ymin_idx, xmax_idx, ymax_idx = range(5)

#imgname_idx, index for image file of tuple
#xmin_idx, index for xmin coordinate of tuple
#ymin_idx, index for ymin coordinate of tuple
#xmax_idx, index for xmax coordinate of tuple
#ymax_idx, index for ymax coordinate of tuple



def MoreThan2Ped(pklfile):
    '''Retrieves image titles from the pkl file that contains more than 
    on pedestrain per image. If only one pedestrian is detected, stores 
    a 1 for the corresponding dictionary value
    
    Parameters
    ----------
    pklfile : 'string'
        .pkl file with bounding box information for pedestrians
          
    Returns
    -------  
    Duplicates : 'dict' 
        Dictionary of all files with the amount duplicates of image titles. 
        Finds the bounding box coordinates of those images. 
        This essentially extracts all images with at least 2 pedestrians.
        If only one pedestrian is detected, stores a 1 for the corresponding 
        dictionary value
    '''
    
    # Column names 
    col_names = ['image','xmin','ymin','xmax','ymax','label','confidence']
    
    # read pkl file     
    data = np.load(pklfile, allow_pickle=True)
    
    # convert into pandas dataframe
    df = pd.DataFrame(data, columns = col_names)
    
    df = df[df['confidence']>= 0.8]
    
    df['image'] = df['image'].str.split('/').str[-1]
    
    images = df['image'].values 
    
    duplicates = {} # keep track of all duplicates 
    
    # Find indices of all duplicates 
    for img in images: 
        if img not in duplicates:
            num_dup = len(np.where(images == img)[0])
            idximg = np.where(images == img)
            if num_dup > 1:
                BoundB = imgBB(idximg, df)
                duplicates[img] = BoundB
            else: 
                duplicates[img] = imgBB(idximg, df)
 
    return duplicates 

def imgBB(idxImg, df):
    '''Finds the bounding boxes of an image given xmin, ymin,
    xmax, ymax vertices 
    
    Parameters
    ----------  
    idxImg : 'np.array'
        Index of images for which you want to find the bounding 
        box coordinates 
    
    df : 'Pandas Dataframe' 
        Dataframe containing all of the bounding box data 
    
    Returns
    -------  
    BoundB: 'np.array'
        All of the bounding box coordinates for this particular image
    '''
    
    cols = ['xmin', 'ymin', 'xmax', 'ymax']
    
    return df.iloc[idxImg][cols].values

def Distance3d(BoundB1, BoundB2, w_actual, fl, Ycam = 80/12, y_horiz = 2750/2): 
    '''Uses pythagorean theorem to calculate the true distance 
    in pixels of two bounding boxes.
    
    Parameters
    ----------  
    BoundB1 : 'np.array'
        Bounding box for 1st pedestrian, contains xmin, xmax, ymin, ymax 
        
    BoundB2 : 'np.array'
        Bounding box for 2nd pedestrian, contains xmin, xmax, ymin, ymax
        
    w_actual : 'float'
        The actual width of a bounding box (e.g. 3 ft or 1 m). This
        is an assumed parameter
        
    fl : 'float'
        Constant focal length in pixels. Also an assumed/calibrated 
        parameter
        
    Ycam : 'float'
        Height of the camera of car in ft 
        
    y_horiz : 'float'
        Estimation of horizon line (middle of image) in pixels 
        
    Output
    ------    
    Distance : 'float'
        Returns an estimated value of distance in ft or m 
    
    '''
    
    y_1 = BoundB1[3] # bottom of bounding box for pedestrian 1 
    y_2 = BoundB2[3] # bottom of bounding box for pedestrian 2
    
    D1 = fl*Ycam/(y_1 - y_horiz) # Distance of pedestrian 1 from car 
    D2 = fl*Ycam/(y_2 - y_horiz) # Distance of pedestrian 2 from car 
    
    # Find pixel width of bounding box
    wpBox1 = BoundB1[2] - BoundB1[0]
    wpBox2 = BoundB2[2] - BoundB2[0]
    
    # Calculate max width between both bounding boxes
    wpBox = max(wpBox1, wpBox2)
    
    # based on width of car in ft and width in pixel
    px_m_ratio = w_actual/wpBox  
    
    # Midpoint of each bounding box 
    mdpoint1 = bb_midpoint(BoundB1)
    mdpoint2 = bb_midpoint(BoundB2)
    
    # Horizontal distance in pixels 
    horiz_dist = abs(mdpoint1[0] - mdpoint2[0])
    
    # Horizontal distance in ft
    w_real = px_m_ratio*horiz_dist
                 
    D = abs(D2-D1)
    
    ActualDist = (D**2 + w_real**2)**0.5
    
    return round(ActualDist,2)

def bb_midpoint(bound_box): 
    '''
    This function calculates the midpoint of a bounding box
    
    Parameters
    ---------- 
    bound_box : 'np.array'
        bound_box[0] = xmin - min x pixel location 
        bound_box[1] = ymin - min y pixel location 
        bound_box[2] = xmax - max x pixel location 
        bound_box[3] = ymax - max y pixel location
    
    Output
    ------ 
        Midpoint of bounding box
    '''
    
    xmin = bound_box[0]
    ymin = bound_box[1]
    xmax = bound_box[2]
    ymax = bound_box[3]
    
    x_mid = (xmax + xmin)/2
    y_mid = (ymax + ymin)/2
    
    return [int(x_mid), int(y_mid)]

def CalcDistances(pklfile, fl = 1364, w_actual = 4): 
    '''
    Calculates all distances between detected pedestrians as inidicated 
    by the pedestrian dectection output pkl file. If only one pedestrian
    is detected in an image, then it will output an arbitrarily large
    distance = 100 ft. 

    Parameters
    ----------
    pklfile : file that contains tuples of all detected pedestrians 
              and their corresponding images
              
    outputfile : file that contains the file name, bounding boxes, 
                and their corresponding combinations of social distances
       
    fl : 'float', optional
        Parameter of focal length. The default is 1364.
        
    w_actual : 'float', optional
        Parameter of assumption for bounding box width in ft. The default is 4.

    Returns
    -------
    None.

    '''
    
    col_names = ['image', 'Bounding Boxes', 
                 'Bound Box Index', 'Social Distances']
    
    dup = MoreThan2Ped(pklfile)
    
    Boundvals = []
    
    for img, boundBs in dup.items():
        if len(boundBs) == 1: 
            Boundvals.append((img, boundBs, [0], [100])) 
        elif len(boundBs) == 2: 
            BoundB1 = boundBs[0]
            BoundB2 = boundBs[1]
            
            dist = Distance3d(BoundB1, BoundB2, w_actual, fl)
            
            Boundvals.append((img, boundBs, [(0, 1)], [dist]))
            
        else: 
            comblist = list(range(len(boundBs)))
            comb = list(combinations(comblist, 2))
            
            
            distances = []
            for combo in comb: 
                BoundB1 = boundBs[combo[0]]
                BoundB2 = boundBs[combo[1]]
                
                dist = Distance3d(BoundB1, BoundB2, w_actual, fl)
                
                distances.append(dist)
                
            Boundvals.append((img, boundBs, comb, distances))   
    
    df = pd.DataFrame(Boundvals, columns = col_names)
    
    return df


def draw_bounding_box(img, bound_box): 
    '''Draws bounding box on image
    
    Input: 
        img - Image that has already been read by imread function
        bounding_box - list = [xmin, ymin, xmax, ymax]
        
    Output: 
        Image with drawn bounding box
    '''
    
    # Top left point of bounding box
    start = (int(bound_box[0]), int(bound_box[1]))
    # Bottom right of bounding box
    end = (int(bound_box[2]), int(bound_box[3]))
    
    # Use red color
    color = (255,0,0)
    
    bb_img = cv2.rectangle(img, start, end, color, 2)
    
    return bb_img

def draw_multiple_box(img, Boxes): 
    '''Draws multiple bounding boxes on image
    
    Input
    -----
    
    img - 'imread file'
        Image that has already been read by imread function
        
    Boxes - 'np.array'
        Multiple bounding boxes in np.array
        
    Output
    ------
    
        Returns annotated image with multiple bounding boxes 
    '''
    
    for i in range(len(Boxes)): 
        img = draw_bounding_box(img, Boxes[i])
        
    return img

def draw_soc_line(img, boundB1, boundB2, distance): 
    '''Draws social distance line between two identified midpoints
    
    Input:   
        img - Image that has already been read by imread function
        
        BoundB1 - 'np.array'
        Bounding box for 1st pedestrian, contains xmin, xmax, ymin, ymax 
        
        BoundB2 - 'np.array'
        Bounding box for 2nd pedestrian, contains xmin, xmax, ymin, ymax
        
        distance - 'float'
            Distance between two bounding boxes in real units (e.g. 6 ft 
                                                               or 2 m)
            
        
    Output: 
        Drawn line representing social distance between two points
        between two identified objects
    '''
    
    mdpoint1 = bb_midpoint(boundB1)
    mdpoint2 = bb_midpoint(boundB2)
    
    start = (mdpoint1[0], mdpoint1[1])
    end = (mdpoint2[0], mdpoint2[1])
    
    # Use red color
    color = (255,0,0)
    
    
    # Find midpoint of midpoints 
    mid_x = int((mdpoint1[0]+ mdpoint2[0])/2)
    mid_y = int((mdpoint1[1]+ mdpoint2[1])/2 - 20)
            
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
      
    # org 
    org = (mid_x, mid_y) 
      
    # fontScale 
    fontScale = 3
       
    # Red color in BGR 
    color = (0, 0, 255) 
      
    # Line thickness of 2 px 
    thickness = 2
    
    if False:
    #if distance <= 6: 
        distanceText = str(distance) + ' ft'
       
        # Using cv2.putText() method 
        
        img = cv2.putText(img, distanceText, org, font, fontScale,  
                         color, thickness, cv2.LINE_AA, False)
        
        soc_line = cv2.line(img, start, end, color, 5) 
        
        return soc_line
    
    elif False:    
    #elif distance > 6 and distance < 20:          
        # Green color in BGR 
        color = (0, 255, 0) 
        
        distanceText = str(distance) + ' ft'
           
        # Using cv2.putText() method 
        
        img = cv2.putText(img, distanceText, org, font, fontScale,  
                         color, thickness, cv2.LINE_AA, False)
        
        soc_line = cv2.line(img, start, end, color, 5) 
        
        return soc_line
    
    else: 
        return img


def AnnotateImg(df_dict, img_name, file_path):
    '''
    This function reads csv file containing images that contain pedestrians 
    bounding box/social distance information and draws bounding boxes and 
    their corresponding social distance to other pedestrians. 
    

    Parameters
    ----------
    df_dict : 'dict'
        Bounding Box and Social Distance information in dictionary form
    photofile : 'str'
        Location of photo directory to be annotated 



    Returns
    -------
    None.

    '''
    
    print(img_name)
    # Read in images
    AnotImg = cv2.imread(file_path)
    
    # Find data for specific image
    img_dat = df_dict[img_name]
    BoundBs = img_dat['Bounding Boxes']
    BB_idxs = img_dat['Bound Box Index']
    SocDist = img_dat['Social Distances']
    
    # Draw Bounding Boxes 
    AnotImg = draw_multiple_box(AnotImg, BoundBs)
    
    
    for BB_idx, dist in zip(BB_idxs, SocDist):
        BB1 = BoundBs[BB_idx[0]]
        BB2 = BoundBs[BB_idx[1]]
        
        AnotImg = draw_soc_line(AnotImg, BB1, BB2, dist)
        
    # Save image
    cv2.imwrite(img_name, AnotImg) 
    
    return None
    
    
   
result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

# test the class
if __name__ == '__main__':
    
    pklfile = PKL_FILE # location of pkl files 
    df = CalcDistances(pklfile, fl = FL, w_actual = W_ACTUAL)
    df.set_index('image', inplace = True)
    df_dict = df.to_dict('index')
    
    outputfile = OUTPUT_DIR # Location of where you want annotated images to be stored 
    os.chdir(outputfile)
    
    rootdir = IMAGE_ROOT_DIR #Location of Original Images to be analyzed 
    pool = mp.Pool(mp.cpu_count())

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fp = os.path.join(subdir, file)
            pool.apply_async(AnnotateImg,args = (df_dict, file, fp),callback= log_result)
 
    pool.close()
    pool.join()
    print(result_list)



