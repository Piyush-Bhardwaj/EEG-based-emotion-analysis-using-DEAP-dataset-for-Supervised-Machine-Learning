def Display_emotion():
    
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    
    eemotion_present=[]
    
    with open('data/emotion_present.dat','r') as f:
        for val in f:
            if str(val) == 'Relaxed\n':
                eemotion_present.append('Relaxed')
            elif str(val) == 'Happy\n':
                eemotion_present.append('Happy')
            elif str(val) == 'Depressed\n':
                eemotion_present.append('Depressed')
            else:
            
                eemotion_present.append('Nervous')
            
        
    f.close()
    
    eemotion_predicted=[]
    
    with open('data/emotion_predicted.dat','r') as f:
        for val in f:
            if str(val) == 'Relaxed\n':
                eemotion_predicted.append('Relaxed')
            elif str(val) == 'Happy\n':
                eemotion_predicted.append('Happy')
            elif str(val) == 'Depressed\n':
                eemotion_predicted.append('Depressed')
            else:
            
                eemotion_predicted.append('Nervous')
            
        
    f.close()
            
               
    img1 = mpimg.imread('image1.jpg')
    img2 = mpimg.imread('image2.jpg')
    img3 = mpimg.imread('image3.jpg')
    img4 = mpimg.imread('image4.jpg')
    
    plt.figure(1)
    
    
    for i in range(len(eemotion_present)):
        plt.subplot(121)
        if eemotion_present[i] == "Happy":
                   
            plt.imshow(img1)
            plt.title("Emotion present")
            plt.xlabel("Happy")
            
        elif eemotion_present[i] == "Nervous":
            plt.imshow(img2)
            plt.title("Emotion present")
            plt.xlabel("Nervous")
            
        elif eemotion_present[i] == "Depressed":
            plt.imshow(img3)
            plt.title("Emotion present")
            plt.xlabel("Depressed")
        
        else:
            plt.imshow(img4)
            plt.title("Emotion present")
            plt.xlabel("Relaxed")
            
             
        plt.subplot(122)
    
        if eemotion_predicted[i] == "Happy":
                   
            plt.imshow(img1)
            plt.title("Emotion predicted")
            plt.xlabel("Happy")
            
        elif eemotion_predicted[i] == "Nervous":
            plt.imshow(img2)
            plt.title("Emotion predicted")
            plt.xlabel("Nervous")
            
        elif eemotion_predicted[i] == "Depressed":
            plt.imshow(img3)
            plt.title("Emotion predicted")
            plt.xlabel("Depressed")
        
        else:
            plt.imshow(img4)
            plt.title("Emotion predicted")
            plt.xlabel("Relaxed")
    
        plt.show()

if __name__ == '__main__':
    Display_emotion()
        
        

