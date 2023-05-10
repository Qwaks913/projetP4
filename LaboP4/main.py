import functions as fn
import os
import platform

if (platform.system() == 'Windows'):
    win = True

def main():
    source_path = os.getcwd()
    save_path = os.getcwd()
    if (win):
        files = os.listdir(f"{source_path}\mesures")
    else:
        files = os.listdir(f"{source_path}/mesures")
    for file_name in files:
        if file_name != ".DS_Store":
            print(f"On s'occupe du fichier suivant: {file_name}")
            if (win):
                file_path = f"{source_path}\mesures\{file_name}"  # Windows
            else:
                file_path =  f"{source_path}/mesures/{file_name}" # Mac et Linux
            #If only load = 0 --> Saves graphs and return chirp matrix
            #If only load = 1 --> Doesn't save graphs and return chirp matrix
            #If only load = 2 --> Doesn't save graphs and returns frame matrix
            #If only load = 3 --> Returns fft's matrix
            fn.fem(file_path, file_name, source_path, only_load=0)
main()
