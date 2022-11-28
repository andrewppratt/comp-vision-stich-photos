# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    overlap_arr = np.eye(N,N)

    def get_homography(img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(gray1,None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(gray2,None)  
        ratio = []
        i=0
        for des1 in descriptors_1:
            ssd = []
            j = 0
            for des2 in descriptors_2:
                diff = np.subtract(des1, des2)
                squares = np.square(diff)
                # store ssd with img1 index(i) img2 index(j)
                ssd.append([i, j, np.sum(squares)])
                j += 1
            #sort the ssd's by ssd
            sorted_ssd = sorted(ssd, key=lambda x: x[2])
            # calculate the ssd ratio with 1st and 2nd values
            ratio.append([sorted_ssd[0][0], sorted_ssd[0][1], sorted_ssd[0][2]/sorted_ssd[1][2]])
            i += 1
        sorted_ratio = sorted(ratio, key=lambda x: x[2], reverse=True)
      
        pts_src = np.float32([ keypoints_1[m[0]].pt for m in sorted_ratio ]).reshape(-1,1,2)
        pts_dst = np.float32([ keypoints_2[m[1]].pt for m in sorted_ratio ]).reshape(-1,1,2)
    
        h, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC,7.0)
        
        return h

    def put_together(im1, im2, h_given=np.zeros((3,3)), shift=False, blend=False):
        if(np.sum(h_given)==0):
            h = get_homography(im1, im2)
        else:
            h = h_given 
        # CALCULATION SIZE OF OUTPUT IMAGE
 
        base_image = im2
        sec_image = im1
        base_im_shape = base_image.shape[:2]
        (height, width) = sec_image.shape[:2]
        initial_mat = np.array([[0, width - 1, width - 1, 0],
                            [0, 0, height - 1, height - 1],
                            [1, 1, 1, 1]])
        final_mat = np.dot(h, initial_mat)
        [x, y, c] = final_mat
        x = np.divide(x, c)
        y = np.divide(y, c)
        min_x, max_x = int(round(min(x))), int(round(max(x)))
        min_y, max_y = int(round(min(y))), int(round(max(y)))
        new_width = max_x
        new_height = max_y
        offset = [0, 0]
        if min_x < 0:
            new_width -= min_x
            offset[0] = abs(min_x)
        if min_y < 0:
            new_height -= min_y
            offset[1] = abs(min_y)

        if new_width < base_im_shape[1] + offset[0]:
            new_width = base_im_shape[1] + offset[0]
        if new_height < base_im_shape[0] + offset[1]:
            new_height = base_im_shape[0] + offset[1]

        x = np.add(x, offset[0])
        y = np.add(y, offset[1])
        old_init_points = np.float32([[0, 0],
                                    [width - 1, 0],
                                    [width - 1, height - 1],
                                    [0, height - 1]])
        new_final_points = np.float32(np.array([x, y]).transpose())
        h = cv2.getPerspectiveTransform(old_init_points, new_final_points)
        new_frame_size = [new_height, new_width] 
        img_out = cv2.warpPerspective(sec_image, h, (new_frame_size[1], new_frame_size[0]))
        img_out_copy = img_out.copy()
        img_out[offset[1]:offset[1]+base_image.shape[0], offset[0]:offset[0]+base_image.shape[1]] = base_image
        # CREATE A MASK TO FIND OUT OVERLAP
        img_out_mask = np.array([[1 if np.sum(pixel) > 0 else 0 for pixel in row] for row in img_out])
        base_img_buf = np.zeros(img_out_mask.shape)
        base_img_mask = np.array([[1 if np.sum(pixel) > 0 else 0 for pixel in row] for row in base_image])
        base_img_buf[offset[1]:offset[1]+base_image.shape[0], offset[0]:offset[0]+base_image.shape[1]] = base_img_mask
        overlap = np.add(img_out_mask, base_img_buf)
        overlap_pixels = np.count_nonzero(overlap == 2)
        if overlap_pixels > 0:
            overlap_percent = overlap_pixels / (np.count_nonzero(overlap == 1) + overlap_pixels)
        else:
            overlap_percent = 0

        #BLENDING - ISH
        if blend:
            zeros = np.zeros(3)
            for i in range(offset[1]-1,offset[1]+base_image.shape[0]):
                for j in range(offset[0]-1,offset[0]+base_image.shape[1]):
                    if (np.sum(img_out[i][j]) == 0):
                        img_out[i][j] = img_out_copy[i][j]
                   

        return img_out, h, overlap_percent

    results = []
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if i == j:
                results.append([])
                continue
            stitch, h, over_pct = put_together(imgs[i], imgs[j])
            if over_pct >= 0.20 and over_pct != 1:
                overlap_arr[i][j] = 1
                results.append([stitch, h, over_pct])
            else:
                results.append([])
            print(overlap_arr)

   
# FIND THE IMAGE WITH THE FEWEST CONNECTIONS
    num_of_cons = []
    for image in overlap_arr:
        num_of_cons.append(np.sum(image)-1)
   
    stitch_120, h012, pct = put_together(results[6][0], imgs[0])
    stitch_1203, h0123, pct = put_together(stitch_120, imgs[3])
    cv2.imwrite(savepath, stitch_1203)

    # for i in range(len(results)):
    #     if len(results[i]) > 0:
    #         pair_name = f'Stitch {int(i/N)} {i%N}'
    #         print(pair_name)
    # cv2.imshow("1203", stitch_1203)

    # cv2.waitKey(0)

    return overlap_arr


if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
   # overlap_arr2 = stitch('t3', savepath='task3.png')
    #with open('t3_overlap.txt', 'w') as outfile:
     #   json.dump(overlap_arr2.tolist(), outfile)
