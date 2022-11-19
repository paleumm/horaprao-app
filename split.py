import splitfolders
input_folder = "./dataset/"
output = "./input/" #where you want the split datasets saved. one will be created if it does not exist or none is set

splitfolders.ratio(input_folder, output=output, seed=1337, ratio=(.8, .2))