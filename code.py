#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import cv2
import numpy as np
from sklearn.feature_extraction import image


# In[4]:
#to get position in list
def getpos(i,j):
    n=i*(const_shape[1]-psize[1]+1)+j
    return n
def getpos2(n):
    row=n/(ims_sp[1]-psize[1]+1)
    col=n%(ims_sp[1]-psize[1]+1)
    return [row,col]

#main function
psize=[7,7]

ims=cv2.imread('./source.jpg')
ims=cv2.resize(ims,(500,500))
#imt=cv2.imread('./test.jpg')
cv2.imwrite('curr_source.jpg',ims)
scale_percent = 90
it=0
width = int(ims.shape[1] * scale_percent / 100)
height = int(ims.shape[0] * scale_percent / 100)

dim = (width, height)
Nt=dim[0]*dim[1]
Ns=(np.shape(ims)[0] * np.shape(ims)[1])
imt = cv2.resize(ims, dim, interpolation = cv2.INTER_AREA)
while(it<8):


   width=imt.shape[1]
   height=imt.shape[0]
   dim = (width, height)
   Nt=dim[0]*dim[1]	#resizing image to desired / req length and width
   p_s = image.extract_patches_2d(ims, (psize[0],psize[1]))

   constant=cv2.copyMakeBorder(imt,psize[0],psize[1],psize[0],psize[1],cv2.BORDER_CONSTANT,value=0)
   p_t = image.extract_patches_2d(constant, (psize[0],psize[1]))



   print(len(p_s))
   print(len(p_t))
   final=np.zeros(np.shape(imt))


# In[5]:


   const_shape=np.shape(constant)
   ims_sp=np.shape(ims)
   print(len(p_t))
   patches_index=[]
   for i in range(len(p_t)):
        if(i%1000 ==0 ):
           print(i)
        match1=cv2.matchTemplate(ims,p_t[i],cv2.TM_SQDIFF)				# using sq distance to be the nearest metric
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match1)		#gives location of best match
        top_left = min_loc
#     bottom_right = (top_left[0] + psize[0], top_left[1] + psize[1])
        patches_index.append(top_left)
   print(len(patches_index))
   imt_sp=np.shape(imt)
   for x in range(psize[0]-1,psize[0]-1+imt_sp[0]):
        for y in range(psize[1]-1,psize[1]-1+imt_sp[1]):
             value=[0,0,0]
             for i in range(x-(psize[0]-1),x):
                 for j in range(y-(psize[1]-1),y):
                     pos=getpos(i,j)
                     [r,c]=patches_index[pos]
                     value[0] += ims[r+x-i,c+y-j,0]
                     value[1] += ims[r+x-i,c+y-j,1]
                     value[2] += ims[r+x-i,c+y-j,2]
             final[y-psize[1]+1,x-psize[0]+1,0]=value[0]/float(Nt)
             final[y-psize[1]+1,x-psize[0]+1,1]=value[1]/float(Nt)
             final[y-psize[1]+1,x-psize[0]+1,2]=value[2]/float(Nt)
        print(x)
   p_t=np.zeros(p_t.shape)
   patches_index[:]=[]
   patches_index2={}
   for i in range(len(p_s)):
         match1=cv2.matchTemplate(constant,p_s[i],cv2.TM_SQDIFF)				# using sq distance to be the nearest metric
         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match1)		#gives location of best match
         top_left = min_loc
#     bottom_right = (top_left)
         if(i%1000 ==0):
            print(i)
         try:
            pos=getpos2(i)
#         print(pos)
            patches_index2[(top_left[0],top_left[1])].append([pos[0],pos[1]])
         except KeyError:
            pos=getpos2(i)
            patches_index2[(top_left[0],top_left[1])]=[ [pos[0],pos[1]] ]
   for x in range((psize[0]-1),(psize[0]-1)+imt_sp[0]):
         for y in range((psize[1]-1),(psize[1]-1)+imt_sp[1]):

            value=[0,0,0]
            for i in range(x-(psize[0]-1),x):
                for j in range(y-(psize[1]-1),y):
                   try:
                     for eachp in patches_index2[ (i,j) ]:
#                         print(eachp)
                        r=int(eachp[0])
                        c=int(eachp[1])
                        value[0] += ims[r+x-i,c+y-j,0]
                        value[1] += ims[r+x-i,c+y-j,1]
                        value[2] += ims[r+x-i,c+y-j,2]

                   except KeyError:
                       pass
            final[y-psize[1]+1,x-psize[0]+1,0]=value[0]/float(Ns)
            final[y-psize[1]+1,x-psize[0]+1,1]=value[1]/float(Ns)
            final[y-psize[1]+1,x-psize[0]+1,2]=value[2]/float(Ns)
         print(x)
   cv2.imwrite('final.jpg',final)


# In[27]:


   import copy
   rmin=final[..., 0].min()
   rmax=final[..., 0].max()

   gmin=final[..., 1].min()
   gmax=final[..., 2].max()

   bmin=final[..., 2].min()
   bmax=final[..., 2].max()
   temp=copy.copy(final)

   temp[:,:,0] *= 255.0/(temp[:,:,0]).max()
   temp[:,:,1] *= 255.0/(temp[:,:,1]).max()
   temp[:,:,2] *= 255.0/(temp[:,:,2]).max()



   cv2.imwrite('test'+str(it)+'.jpg',temp)
   imt=temp
