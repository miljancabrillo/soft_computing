import numpy as np
import matplotlib
import cv2 # OpenCV
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from number import Number
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 16,12



def distance(a,b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def is_between(a,c,b):
    return -0.09<(distance(a,c) + distance(c,b) - distance(a,b))<0.09
   
def find_lines(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    ret, frame_binary = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 2))
    frame_eroded = cv2.erode(frame_binary, kernel, iterations=3)
   
    
    
    lines = cv2.HoughLinesP(frame_eroded,rho = 1,theta = 1*np.pi/180,threshold = 160,minLineLength = 150,maxLineGap = 10)
    
    first_line = lines[0]
    second_line = None
    #print(lines)
 
    for line in lines[1::]:
        x1,x2,x3,x4 = first_line[0]
        y1,y2,y3,y4 = line[0]
        difference = (abs(x1-y1) + abs(x2-y2) + abs(x3-y3) + abs(x4-y4))/4
        if difference > 20:
            second_line = line
            break
            return
    x1,x2,x3,x4 = first_line[0]
    y1,y2,y3,y4 = second_line[0]
    
    upper_line = None
    lower_line = None
    
    if((x2+x4) < (y2+y4)):
        upper_line = first_line[0]
        lower_line = second_line[0]    
    else:
        upper_line = second_line[0]
        lower_line = first_line[0]
        
    return[upper_line, lower_line]
        
def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255
def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()
def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann
def prepare_outputs_for_ann(outputs):
    #za svaki podataka iz y_train kreiram niz od 10 elementa sa 1 na odgovarajucem mjestu
    ann_outputs = []
    for number in outputs:
        output = np.zeros(10)
        output[number] = 1
        ann_outputs.append(output)
    return np.array(ann_outputs)
def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32) # dati ulazi
    y_train = np.array(y_train, np.float32) # zeljeni izlazi za date ulaze  
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=500, batch_size=1, verbose = 0, shuffle=False) 
      
    return ann

def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    alphabet = [0,1,2,3,4,5,6,7,8,9]
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def select_roi(image_orig, image_bin,line):
    
    x1,y1,x2,y2 = line
    a = [x1,y1]
    b = [x2,y2]
    
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    number_contours = []
    new_numbers = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        t = [x-15, y-15]        
        if x>5:
            x=x-4
        if y>5:
            y=y-4
      
        if ((w > 1 and h > 10) or (w>14 and h>5)) and (w<=28 and h<=28):
            #ako pripada brojevima provjeravam
            #print(is_between(a,t,b))
            if not is_between(a,t,b):
                continue
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            number_contours.append(contour)
            region = image_bin[y:y+h+8,x:x+w+8]
            coords = [x,y,w,h]
            # regions_array.append([resize_region(region), (x,y,w,h)])  
            r = resize_region(region)
            regions_array.append(r) 
            num = Number(coords,r)
            new_numbers.append(num)
            cv2.rectangle(image_orig,(x,y),(x+w+10,y+h+10),(0,255,0),2)
    # sortirati sve regione po  osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, regions_array,number_contours,new_numbers

def update_numbers_array(old_numbers, new_numbers):
    for new_number in new_numbers:
        for old_number in old_numbers:
            x1_old,y1_old,w_old,h_old = old_number.coordinates
            x1_new,y1_new,w_new,h_new = new_number.coordinates
            if distance([x1_old,y1_old],[x1_new,y1_new]) < 2.5:
                new_number.calculated = True
                old_number.coordinates = new_number.coordinates
    return new_numbers

file= open("out.txt","w+")
file.write("RA 121/2015 Miljan Cabrilo\r")
file.write("file	sum\r")

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
ann = model_from_json(model_json)
ann.load_weights("model.h5")

i=0
for i in range(0,10):
    cap = cv2.VideoCapture('data/video-'+str(i)+'.avi')
    totalSum = 0

   

    if cap.isOpened():
        ret,firstFrame = cap.read()
    
    upper_line, lower_line = find_lines(firstFrame)  

    upper_line_numbers =   []
    lower_line_numbers =   []
   
    frames_num = 0
  
    while cap.isOpened():
        
        ret,frame = cap.read()

        if frame is None:
            break
        
        frames_num +=1
        
        #if frames_num == 60:
          #  break
        
        #if not frames_num%5 == 0:
        #    continue
        #print(frames_num)
        #pretvorim sliku u binarno i obradim je otvaranjem i erozijom
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); 
        ret, frame_binary = cv2.threshold(frame_gray, 222, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS , (2, 2))
      
        frame_binary_closed = cv2.dilate(frame_binary, kernel, iterations=2)
        frame_binary_closed = cv2.erode(frame_binary_closed, kernel1, iterations=2)
        
        img1, regions_upper_line, contours1, new_nums_upper = select_roi(frame.copy(),frame_binary_closed.copy(),upper_line)
        img2, regions_lower_line, contours1, new_nums_lower = select_roi(frame.copy(),frame_binary_closed.copy(),lower_line)
      
        #plt.figure()
        #cv2.imshow('Frame',img1)
        
        if not(not regions_upper_line):
            upper_line_numbers = update_numbers_array(upper_line_numbers,new_nums_upper)
            
            for number in upper_line_numbers:
                if number.calculated == False:
                    num = ann.predict(np.array(prepare_for_ann([number.region]),np.float32))
                    display_result(num)
                    totalSum += sum(display_result(num))
                    number.calculated = True
            #print('upper')
            #print(display_result(result1))
           
    
        if not(not regions_lower_line):
                lower_line_numbers = update_numbers_array(lower_line_numbers,new_nums_lower)
                
                for number in lower_line_numbers:
                    if number.calculated == False:
                        num = ann.predict(np.array(prepare_for_ann([number.region]),np.float32))
                        totalSum -= sum(display_result(num))
                        number.calculated = True
                #print('upper')
                #print(display_result(result1))
            
       
        
        #print(totalSum)
        #for im in regions:
           # plt.figure()
            #plt.imshow(im,'gray')

    
        #plt.imshow(img) 
        #result = ann.predict(np.array(prepare_for_ann(regions),np.float32))
        #print(display_result(result))
   
    file.write('video-'+str(i)+'.avi\t' + str(totalSum)+'\r')
    print(totalSum)
    #print(frames_num)
    cv2.waitKey(5000)
    cap.release()
    cv2.destroyAllWindows() 
file.close()