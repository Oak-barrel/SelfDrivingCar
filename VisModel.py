
import sys
import numpy as np
import h5py
import pygame
import json
#from keras.models import model_from_json

H,W = 320, 160
#H,W = 64,32
pygame.init()
size = (H*2, W*2)
pygame.display.set_caption("comma.ai data viewer - adapted")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

camera_surface = pygame.surface.Surface((H,W),0,24).convert()

import scipy.misc as misc

# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

def perspective_tform(x, y):
  p1, p2 = tform3_img((x,y))[0]
  return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
  row, col = perspective_tform(x, y)
  if row >= 0 and row < img.shape[0] and\
     col >= 0 and col < img.shape[1]:
    img[row-sz:row+sz, col-sz:col+sz] = color

def draw_path(img, path_x, path_y, color):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color)

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255)):
  path_x = np.arange(0., 50.1, 0.5)
  path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
  draw_path(img, path_x, path_y, color)



# ***** main loop *****
if __name__ == "__main__":
    import sys
    fname = sys.argv[1]
    #fname = '2016-01-31--19-19-25.h5'
    #fname = '2016-01-30--11-24-51.h5'
    #fname = '2016-03-29--10-50-20.h5'
    dataname = '../datasets/Modified-%s' %(fname)
    data = h5py.File(dataname,'r')
    #data = h5py.File('../datasets/Modified-2016-04-21--14-48-08.h5','r')
    #data = h5py.File('../datasets/Modified-2016-02-08--14-56-28.h5','r')
    #cam = h5py.File("../datasets/2016-04-21--14-48-08.h5","r")
    #log = h5py.File("../datasets/log-2016-04-21--14-48-08.h5","r")
    #log = h5py.File('../datasets/log-2016-02-08--14-56-28.h5','r')
    clock = pygame.time.Clock()

    milliseconds = clock.tick(20)
    #print log.keys()
    skip_frames = 2 # 10 => 1 per sec
    for i in xrange(0*data['X'].shape[0]/10,data['X'].shape[0], skip_frames):
        print "%.2f seconds elapsed" % (i/100.0)
        #imgsmall = cam['X'][log['cam1_ptr'][i]].swapaxes(0,1)
        #img = cam['X'][log['cam1_ptr'][i]].swapaxes(0,2).swapaxes(0,1)
        imgsmall = data['X'][i].swapaxes(0,1).astype(np.uint8)
        print imgsmall.shape
        img = (misc.imresize(imgsmall, (160,320,3))).astype(np.float32)
        #predicted_steers = model.predict(img[None, :, :, :].transpose(0, 3, 1, 2))[0][0]
        predicted_steers = 0.0

        angle_steers = data['steering_angle'][i]
        speed_ms = data['speed'][i]
        print angle_steers, predicted_steers
        draw_path_on(img, speed_ms, -angle_steers/10.0)
        draw_path_on(img, speed_ms, -predicted_steers/10.0, (0, 255, 0))
        # draw on
        pygame.surfarray.blit_array(camera_surface, img.swapaxes(0,1))
        camera_surface_2x = pygame.transform.scale2x(camera_surface)
        screen.blit(camera_surface_2x, (0,0))
        text = "FPS: {0:.2f}   Playtime: {1:.2f}".format(clock.get_fps(), i/100)
        pygame.display.set_caption(text)
        #pygame.image.save(camera_surface_2x, '%d.jpg'%(i/100))
        pygame.display.flip()
        #break

