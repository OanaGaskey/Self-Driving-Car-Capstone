import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image 

def object_extract(image):   
    bbox, label, conf = cv.detect_common_objects(image)
    output_image = draw_bbox(image, bbox, label, conf)
    plt.imshow(output_image)
    
    # Setting the points for cropped image 
    try:
        left = bbox[0][0]
        top = bbox[0][1]
        right = bbox[0][2]
        bottom = bbox[0][3]
        
    except IndexError:
        left = bbox[1][0]
        top = bbox[1][1]
        right = bbox[1][2]
        bottom = bbox[1][3]
    
    # Cropped image of above dimension 
    # (It will not change orginal image) 
    
    output_image = Image.fromarray(output_image)
    output_image = output_image.crop((left, top, right, bottom)) 
    
    return output_image
