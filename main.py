import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import os
from flask import Flask, render_template, request
import base64
from PIL import Image

import flask
app = Flask(__name__)


def calculate_DCP(img, wind_size):
  """
  calculates the dark channel prior for the input image
  it select the pixel having low intensity within the patch
  cv2.copyMakeBorder() method is used to create a border around the image like a photo frame
  cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
  """
  dcp = np.zeros((img.shape[0], img.shape[1]))
  border_size = wind_size//2
  img = cv2.copyMakeBorder(img, 
                           border_size,
                           border_size,
                           border_size,
                           border_size,
                           cv2.BORDER_CONSTANT, 
                           value=[255, 255, 255])
  num_rows = img.shape[0]
  num_cols = img.shape[1]
  min_channel = np.zeros((num_rows, num_cols))

  for row in range(num_rows):
    for col in range(num_cols):
      min_channel[row-border_size][col-border_size] = np.min(img[row, col, :]) #finds the minimum intensity in each channel

  for row in range(border_size, num_rows-border_size):
    for col in range(border_size, num_cols-border_size):
      dcp[row-border_size][col-border_size] = np.min(min_channel[row-border_size:row+border_size, col-border_size:col+border_size]) #calculates dark channel prior
      
  return dcp

def calculate_ambience(im, dc_img):
  """
  Returns atmospheric ambience(brightness paramater "A")
  input -
  im  = original Image
  dc_img = dark channel prior of the orirginal image
  """
  img = im.copy()
  pixel_count = dc_img.size
  # print(pixel_count)
  count_brightest = pixel_count//1000  # pick the top 0.1% brightest pixels in the dark channel
  haze_density_sort_idx = np.argsort(dc_img, axis=None)[::-1]
  brightest = haze_density_sort_idx[0:count_brightest]
  brightest = np.unravel_index(brightest,dc_img.shape)
  brightest_pixels = img[brightest]
  top_intensities = np.average(brightest_pixels, axis=1)
  max_intensity = np.argmax(top_intensities) #finds maximum among the brightest pixel
  A = brightest_pixels[max_intensity]

  # to display the brightest pixels in the image
  img[brightest]=[255,0,0]
  row_min = np.min(brightest[0])
  row_max = np.max(brightest[0])
  col_min = np.min(brightest[1])
  col_max = np.max(brightest[1])
  
  # mark the brightest region in the image
  cv2.rectangle(img, 
                (col_min,row_min),
                (col_max,row_max),
                (0,0,255),
                thickness=2)
  
  # plt.figure(figsize=(10,10))
  # plt.imshow(img[...,::-1])
  # plt.show()
  return A

def filter_image(img, transmission, filter_size, epsilon):
  """
  smoothen the input image(transmssion map) using guided filter.
  input -
  img = original image
  transmission = transmission t(x)
  filter_size = width of the guided filter
  epsilon  =  constant value

  output -
  q = egde preserved smoothen image

  """
  guide = cv2.blur(img,(filter_size,filter_size)) #smoothen the guiding image
  trans = cv2.blur(transmission,(filter_size,filter_size)) # smoothen the transmission map
  gt = cv2.blur(img * transmission, (filter_size,filter_size))
    
  a = gt - guide * trans
  var_guide = cv2.blur(img * img,(filter_size,filter_size)) - (guide *guide)
  a = a/(var_guide + epsilon)
  b = trans - a * guide

  q = cv2.blur(a,(filter_size,filter_size)) * img + cv2.blur(b,(filter_size,filter_size))
  return q

def recover_image(img, trans_bar, atm_light, t0):
  """
  recover original image
  input -
  img = original image
  trans_bar = transmission map for hazed image
  atm  = atmospheric brightness A
  t0 = lower bound for t(x)

  output -
  j = dehazed image


  """
  trans_recover = np.copy(trans_bar)
  trans_recover[trans_recover < t0] = t0
  J = np.zeros((img.shape))

  J[:,:,0] = ((img[:,:,0] - atm_light[0])/trans_recover) + atm_light[0] #recovery got R channel
  J[:,:,1] = ((img[:,:,1] - atm_light[1])/trans_recover) + atm_light[1] #recovery for G channel
  J[:,:,2] = ((img[:,:,2] - atm_light[2])/trans_recover) + atm_light[2] #reovery for B channel
  
  return J

def color_balance(img, s):
  """
  since the haze removal also affects the brightness
  it enhance the color saturation of the dehazed image
  """
  out = np.copy(img)
  hist = np.zeros((256,1))
  no_of_pixels = img.shape[0] * img.shape[1]

  for i in range(3):
    channel_vals = img[:,:,i]

    for pixel_val in range(256):
      hist[pixel_val] = np.sum((channel_vals == pixel_val)) 
    for pixel_val in range(256):
      hist[pixel_val] = hist[pixel_val-1] + hist[pixel_val]

    # clipping pixels
    Vmin = 0
    while (Vmin < 255 and hist[Vmin] <= no_of_pixels*s):
      Vmin += 1
    Vmax = 255
    while (Vmax > 0 and hist[Vmax] > no_of_pixels*(1-s)):
      Vmax -= 1
    channel_vals[channel_vals < Vmin] = Vmin
    channel_vals[channel_vals > Vmax] = Vmax

    # normalize pixel values
    out[:,:,i] = cv2.normalize(channel_vals, channel_vals.copy(), 0, 255, cv2.NORM_MINMAX)
  return out

def depth_map(t_refine, beta):
      x = -np.log(t_refine)/beta
      return x

@app.route('/')
def render_webpage():
     return flask.render_template("image.html")
     
@app.route('/send', methods=['POST'])
def dehazed_image():
    ret_string = ''
    if request.method == 'POST':
        base = request.form.get('image')
        comma = base.index(',')
        base = base[comma + 1::]
        image = base64.b64decode(base)
        fh = open("inputimage.jpg", "wb")
        fh.write(image)
        fh.close()
        im = Image.open("inputimage.jpg")
        img = cv2.imread('inputimage.jpg')
        img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
        dcp_img = calculate_DCP(img, 15)
        dcp_img = dcp_img.astype('uint8')
        atm_light = calculate_ambience(img, dcp_img)
        t_bar = calculate_DCP(img/atm_light,15)
        trans_bar = 1-(0.85*t_bar)
        i=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)/255
        t_refine = filter_image(i, trans_bar, 30, 0.0001)
        im = img.astype("double")
        J = recover_image(im, t_refine, atm_light, 0.1)
        J = ((J-np.min(J))/(np.max(J)-np.min(J)))*255
        cb_J = color_balance(np.uint8(J),0.005)
        return_image = np.uint8(cb_J[...,::-1])
        plt.imsave("dehazed.jpg", return_image)
        with open("dehazed.jpg", "rb") as f:
            ret_string = base64.b64encode(f.read())
            ret_string = "data:image/jpeg;base64," + ret_string.decode("utf-8")
        f.close()
    return ret_string, 200



if __name__ == "__main__":
    app.run()