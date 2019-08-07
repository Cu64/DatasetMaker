import logging, cv2, os, re
from PIL import Image, ImageStat
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d/%m/%y %H:%M:%S')

def get_images(directory):
    """
    Find all image files in the directory specified.

    :directory: Path to the directory
    """
    images = []
    for file in os.listdir(directory):
        if file.endswith(".jpg") or file.endswith(".png"):
            logging.info(f"Found image {file}")
            images.append(file)
        else:
            logging.info(f"Unrecognized file {file}")
    return images

def filter_images(directory, images):
    """Filter images in the array specified
    
    :directory: the directory that the images array was generated from

    :images: the array that was generated from get_images()
    """
    for image in images:
        try:
            img = Image.open(directory + '\\' + image)
            bands = img.getbands()
            if bands == ('R','G','B') or bands== ('R','G','B','A'):
                logging.info(f"Image {image} is a colored image")
                pass
            elif len(bands)==1:
                logging.info(f"Image {image} is not a colored image, skipping")
                images.remove(image)
            else:
                logging.info(f"Image {image} is unrecogniseable")
        except OSError:
            logging.error(f"Image {image} has invalid exif segment")
    return images

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """
    Resize image

    :image: The image that needs to be resized

    :width: Width of the image after resizing

    :height: Height of the image after resizing
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    logging.info(f"Cropped 1 image")
    return resized

def detect_faces(directory, images, cascadefile):
    """
    Detect faces from images from the image array provided

    :directory: directory that contain the images

    :images: array of images

    :cascadefile: path to cascade file
    """
    if not os.path.isfile(cascadefile):
        raise RuntimeError("%s: not found" % cascadefile)

    for image in images:
        cascade = cv2.CascadeClassifier(cascadefile)
        img = cv2.imread(directory + '\\' + image, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray,
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (64, 64))
        logging.info("Found " + str(len(faces)) + f" faces in {image}") 
        for (x, y, w, h) in faces:
            cropped = img[y:y+h, x:x+w]
            resized = image_resize(cropped, width=64, height=64)
            cv2.imwrite('data/' + image.replace('.jpg', '').replace('.png', '') + '.png', resized)

images = get_images("D:\GrabberData")
colored_images = filter_images("D:\GrabberData", images)
#colored_images = ['0b9101869c35905c19209068cb99f37d.jpg']
detect_faces("D:\GrabberData", colored_images, "cascade/lbpcascade_animeface.xml")