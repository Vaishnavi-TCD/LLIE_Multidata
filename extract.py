import rarfile

rar = rarfile.RarFile("Z:/Documents/Low_light_image_restoration/Datasets/Unpaired_Dataset/LIME.rar")
rar.extractall(path="Z:/Documents/Low_light_image_restoration/Datasets/Unpaired_Dataset/LIME")
rar.close()
