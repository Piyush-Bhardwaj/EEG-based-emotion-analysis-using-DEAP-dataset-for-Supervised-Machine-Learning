def onehotencoding1():
    print("Program started"+"\n")
    fout_labels_class = open("data/label_class_1.dat",'w')
    
    with open('data/labels_1.dat','r') as f:
        for val in f:
            if float(val) > 4.5:
                fout_labels_class.write(str(1) + "\n");
            else:
                fout_labels_class.write(str(0) + "\n");
    print("Encoded label 1"+"\n")
if __name__ == '__main__':
    onehotencoding1()