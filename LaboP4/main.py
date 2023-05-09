import functions as fn
import os
import platform
def main():
    source_path = os.getcwd()
    save_path = os.getcwd()
    if (platform.system() == 'Windows'):
        files = os.listdir("%s\mesures" % source_path)
    else:
        files = os.listdir("%s/mesures" % source_path)
    for file_name in files:
        if file_name != ".DS_Store":
            print("On s'occupe du fichier suivant: " + file_name)
            if (platform.system() == 'Windows'):
                file_path = "%s\mesures\%s" % (source_path, file_name)  # Windows
            else:
                file_path = "%s/mesures/%s" % (source_path, file_name)  # Mac et Linux
            #If only load = 0 --> Saves graphs and return chirp matrix
            #If only load = 1 --> Doesn't save graphs and return chirp matrix
            #If only load = 2 --> Doesn't save graphs and returns frame matrix
            #If only load = 3 --> Returns fft's matrix
            fn.fem(file_path, file_name, source_path, only_load=0)
measure_file = 'Tracking-40.npz'
#fn.angle(measure_file)
main()
