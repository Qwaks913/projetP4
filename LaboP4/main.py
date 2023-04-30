import functions as fn
import os
import platform
def main():
    source_path = os.getcwd()
    save_path = os.getcwd()
    if (platform.system() == 'Windows'):
        files = os.listdir("%s\\mesures" % source_path)
    else:
        files = os.listdir("%s/mesures" % source_path)
    for file in files:
        if file != ".DS_Store":
            print(file)
            if (platform.system() == 'Windows'):
                file2 = "%s\\mesures\\%s" % (source_path, file)  # Windows
            else:
                file2 = "%s/mesures/%s" % (source_path, file)  # Mac et Linux
            #If only load = 0 --> Saves graphs and return chirp matrix
            #If only load = 1 --> Doesn't save graphs and return chirp matrix
            #If only load = 2 --> Doesn't save graphs and returns frame matrix
            fn.fem(file2, file, source_path, only_load=0)
measure_file = 'GR13_4m_2l.npz'
#fn.angle(measure_file)
main()