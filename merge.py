import os

label_path="OCR/training_data/annotations"
label_list=os.listdir(label_path)
label_list=[i for i in label_list if i.endswith(".txt")]
list_data=[]
for label in label_list:
    label_file=os.path.join(label_path,label)
    data=open(label_file,"r",encoding="utf-8").readlines()
    data=[i.strip() for i in data]
    list_data.extend(data)
with open("data.txt","w",encoding="utf-8") as f:
    for line in list_data:
        f.write(line+"\n")
