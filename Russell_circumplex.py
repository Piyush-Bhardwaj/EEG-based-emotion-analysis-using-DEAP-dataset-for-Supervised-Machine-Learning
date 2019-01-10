

def Russell_circumplex():
    yy_test_0=[]
    
    with open('data/y_test_0.dat','r') as f:
        for val in f:
            
            if str(val) == '0.0\n':
                yy_test_0.append(0)
            if str(val) == '1.0\n':
                yy_test_0.append(1)
                
    f.close()
        
    yy_test_1=[]
    
    with open('data/y_test_1.dat','r') as f:
        for val in f:
            if str(val) == '0.0\n':
                yy_test_1.append(0)
            if str(val) == '1.0\n':
                yy_test_1.append(1)
    f.close()
        
    yy_predict_0=[]
    
    with open('data/y_predict_0.dat','r') as f:
        for val in f:
            if str(val) == '0.0\n':
                yy_predict_0.append(0)
            if str(val) == '1.0\n':
                yy_predict_0.append(1)
    f.close()
        
    yy_predict_1=[]
    
    with open('data/y_predict_1.dat','r') as f:
        for val in f:
            if str(val) == '0.0\n':
                yy_predict_1.append(0)
            if str(val) == '1.0\n':
                yy_predict_1.append(1)
            
        f.close()
        
            
    
    print("\n Program Started for Russell circumplex ")
    emotion_present=[]
    for i in range(len(yy_test_0)):
        if yy_test_0[i] == 0 and yy_test_1[i] == 0:
            emotion_present.append("Depressed")
        elif yy_test_0[i] == 0 and yy_test_1[i] == 1:
            emotion_present.append("Nervous")
            
        elif yy_test_0[i] == 1 and yy_test_1[i] == 0:
            emotion_present.append("Relaxed")
            
        else:
            emotion_present.append("Happy")
    print("\n Actual Emotion Present:  ")
            
    print(emotion_present)
    
    eemotion_present_file = open("data/emotion_present.dat",'w')
    for i in range(len(yy_test_0)):
        eemotion_present_file.write(str(emotion_present[i]) + '\n')
    eemotion_present_file.close()
    
       
    emotion_predicted=[]
    for i in range(len(yy_predict_0)):
        if str(yy_predict_0[i]) == 0 and str(yy_predict_1[i]) == 0:
            emotion_predicted.append("Depressed")
        elif yy_predict_0[i] == 0 and yy_predict_1[i] == 1:
            emotion_predicted.append("Nervous")
            
        elif yy_predict_0[i] == 1 and yy_predict_1[i] == 0:
            emotion_predicted.append("Relaxed")
            
        else:
            emotion_predicted.append("Happy")
    print("\n Emotion Predicted:  ")
            
    print(emotion_predicted)
    
    eemotion_predicted_file = open("data/emotion_predicted.dat",'w')
    for i in range(len(yy_test_0)):
        eemotion_predicted_file.write(str(emotion_predicted[i]) + '\n')
    eemotion_predicted_file.close()
    
if __name__ == '__main__':
    Russell_circumplex()
    
        
            
            
        
        
    