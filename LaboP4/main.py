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
            fn.fem(file2, file, source_path, graph=True, only_load=False)
fn.angle2()