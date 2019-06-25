#makes new letters from the old ones
import os
#from combination import combination
#from crop import crop
from noise import noise
from rotation import rotation
#from scaling import scaling
from shearing import shearing
from translation import translation
import cv2


types = ['combination', 'crop', 'noise', 'rotation', 'scaling', 'shearing', 'translation']

path_here = os.path.dirname(os.path.realpath(__file__))
path_one_up = os.path.dirname(path_here)
path_new_letters = path_one_up + '/new_letters'

if not os.path.exists(path_new_letters):
    os.mkdir(path_new_letters)
path_original_letters = path_one_up + '/original_letters' 
letternames= os.listdir(path_original_letters)

for letter in letternames:
    letterpath = path_original_letters + '/' + letter
    newletterpath = path_new_letters + '/' + letter + '/'
    if not os.path.exists(newletterpath):
        os.mkdir(newletterpath)

            
    imagenames = os.listdir(letterpath)
    for imagename in imagenames[0:5]:
        imagepath = letterpath + '/' + imagename
        image = cv2.imread(imagepath,0)           #read as grayscale image
        #print(newletterpath)
        #Now we will do the follow with all images: 
        rotation(image, newletterpath + imagename[0:-4], [5,10,15,20,25,30])
        noise(image, newletterpath + imagename[0:-4], [1.2,1.4,1.6,1.8,2])
        translation(image, newletterpath + imagename[0:-4],[3,5])
        shearing(image, newletterpath + imagename[0:-4],[0,0.5,0.8,1.2])
        
        
        
        
        
        

        
        
        

        
         
    
    

