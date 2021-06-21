"""

Example usage:

python3 CoralAI_detector.py \
  --model ./model/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ./model/coco_labels.txt \

"""

import argparse
import time
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
import cv2 as cv
from imutils.video import FPS
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)



def draw_objects(draw, objs, labels, depth_image):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    
    if labels.get(obj.id, obj.id) == 'person':
      draw.text((bbox.xmin + 10, bbox.ymin + 10),
                '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                fill='red')
      draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                    outline='red')
      bbox_xc = round(bbox.xmin + (bbox.xmax - bbox.xmin)/2)
      #print(bbox_xc)
      bbox_yc = round(bbox.ymin + (bbox.ymax - bbox.ymin)/2)
      #print(bbox_yc)
      #print(depth_image[bbox_yc, bbox_xc])
      draw.text((bbox.xmin + 10, bbox.ymin + 90),
            'DISTANCE[m]\n%.2f' % (round(depth_image[bbox_yc, bbox_xc]/1000,1)),
            fill='red')

  


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file')
  parser.add_argument('-i', '--input', required=False,
                      help='File path of image to process')
  parser.add_argument('-l', '--labels', help='File path of labels file')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  args = parser.parse_args()

  labels = read_label_file(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  
  # Start streaming
  pipeline.start(config)
    

  # Define the codec and create VideoWriter object
  fourcc = cv.VideoWriter_fourcc(*'XVID')
  out = cv.VideoWriter('output1.avi', fourcc, 20.0, (640,  480))


  # start the frames per second throughput estimator
  fps = FPS().start()

  count = 0

  while True:

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
      continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    #print(depth_image.shape)
    #print(depth_image[240, 320])
    color_image = np.asanyarray(color_frame.get_data())
    frame = color_image

    
    #convert frame to PIL image
    image = Image.fromarray(frame)
    interpreter.invoke()
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

 
    objs = detect.get_objects(interpreter, args.threshold, scale)

   

    image = image.convert('RGB')
    #print(image.shape)
    draw_objects(ImageDraw.Draw(image), objs, labels, depth_image)
    #image.save(args.output)
    frame = np.array(image)

    #save frames into video
    out.write(frame)


    #print(frame.shape)
    cv.imshow("Frame", frame)

    # update the FPS counter
    fps.update()


    #image.show() 
    key = cv.waitKey(1) & 0xFF     

    if key == ord('q'):
        break


    count += 1
    if count == 1500:
        break
    


  # stop the timer and display FPS information
  fps.stop()
  print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


  # When everything done, release the capture  
  out.release()
  cv.destroyAllWindows()

  pipeline.stop()


if __name__ == '__main__':
  main()
