import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import platform
from cmath import polar, exp, phase, rect
def angle():
    roller = False 
    delete_dc = True
    source_path = os.getcwd()
    c = 3*10**8
    lam = c/(2.4*10**9)
    k = 2*np.pi/lam
    file = 'calib0.npz'
    if (platform.system() == 'Windows'):
        name_cal = "%s\\calibration_file\\%s" % (source_path, file)  # Windows
    else:
        name_cal = "%s/calibration_file/%s" % (source_path, file)  # Mac et Linux
    #mesure
    file = 'GR13_4m_4l.npz'
    if (platform.system() == 'Windows'):
        name_file = "%s\\mesures\\%s" % (source_path, file)  # Windows
    else:
        name_file = "%s/mesures/%s" % (source_path, file)  # Mac et Linux
    I1_cal, Q1_cal, I2_cal,Q2_cal,Ns_cal = fem(name_cal)
    I1_mes, Q1_mes, I2_mes, Q2_mes,Ns_mes = fem(name_file)

    #Reconstitution des signaux

    cal1 = I1_cal - complex(0,-1)*Q1_cal
    if (delete_dc):
        for i in range(len(cal1)):
            cal1[i:i+Ns_cal] = cal1[i:i+Ns_cal] - np.mean(cal1[i:i+Ns_cal])

    cal2 = I2_cal - complex(0,-1)*Q2_cal
    if (delete_dc):
        for i in range(len(cal1)):
            cal2[i:i+Ns_cal] = cal2[i:i+Ns_cal] - np.mean(cal2[i:i+Ns_cal])
    mes1 = I1_mes - complex(0, -1) * Q1_mes
    if(delete_dc):
        for i in range(len(mes1)):
            mes1[i:i+Ns_mes] = mes1[i:i+Ns_cal] - np.mean(mes1[i:i+Ns_cal])
    mes2 = I2_mes - complex(0, -1) * Q2_mes
    if (delete_dc):
        for i in range(len(cal1)):
            mes2[i:i+Ns_cal] = mes2[i:i+Ns_cal] - np.mean(mes2[i:i+Ns_cal])
    diff_phase = np.angle(cal1) - np.angle(cal2)
    alpha = np.mean(diff_phase)
    #alpha = phase(np.mean(cal1))-phase(np.mean(cal2))
    #différence de phase sur tous les points et faire la moyenne de cette différence
    mes2 = mes2*exp(1j*alpha)

    for i in range(len(mes1)):
        diff_phase = np.angle(mes1[i]) - np.angle(mes2[i])
        alpha = abs(np.mean(diff_phase))
        angle = (np.degrees(np.arccos((alpha / np.pi)))) - 90
        print(angle)








def fem(name,base_name = None,source_path = None,graph = None,only_load = True):

    #name = 'GR13_MES1_11m.npz'
    with np.load(name, allow_pickle=True) as mes:
        #On importe les données (si les colonnes existent)
        data = mes['data']
        if 'data_times' in mes:
            data_times = mes['data_times']
        background = mes['background']
        if 'background_times' in mes:
            background_times = mes['background_times']
        chirp = mes['chirp']
        f0 = chirp[0] #Fréquence de la porteuse
        B = chirp[1] #
        Ns = int(chirp[2]) #Le nombre de points par chirp (utiles uniquement)
        Nc = int(chirp[3]) #Le nombre de chirp dans une frame
        Ts = (chirp[4]) #Période d'échantillonage
        Tc = (chirp[5]) #Période de répétition des chirps en seconde
        if 'datetime' in mes:
            datetime = mes['datetime']

        N_frame = len(data)
        c = 3*10**8
        #On print les données
        print("B = "+str(B))
        print("Ms = " + str(Ns))
        print("Fs = " + str(1/Ts))
        print("f0 = " + str(f0))
        print("Tc = " + str(Tc))
        print("Nc = " + str(Nc))

        to_Throw = int(len(data[0][0]) - Ns * Nc) #nombre total de points à supprimer par frame
        to_Throw_chirp = int(to_Throw / Nc) #Nombre de points à supprimer par chirp
        newLen = int(len(data[0][0])-to_Throw) #définit la taille d'un frame après avoir retiré les points de pause
        usefull  =1- to_Throw_chirp/Ns
        I_1 = np.zeros((len(data), newLen))
        Q_1 = np.zeros((len(data), newLen))
        I_2 = np.zeros((len(data), newLen))
        Q_2 = np.zeros((len(data), newLen))



#On supprime les points de pauses
        for i in range(len(data)):
            indexes = []
            for j in range (Ns, (Ns+to_Throw_chirp) * Nc, Ns + to_Throw_chirp):
                for k in range(j, j + to_Throw_chirp):
                    indexes.append(k)
            I_1[i] = np.delete(data[i][0],indexes)
            I_2[i] = np.delete(data[i][2],indexes)
            Q_1[i] = np.delete(data[i][1],indexes)
            Q_2[i] = np.delete(data[i][3],indexes)

        #l'array data contient N_frames frames contenant chacuns les points recues par les différentes antennes
        if(only_load):
            return (I_1,I_2,Q_1,Q_2,Ns)

        #Nous disposons maintenant des données recueillies par l'antenne sans les pauses.
        full_signal = I_1+complex(0,-1)*Q_1 #on additionne les parties réelles et immaginaires
        #A ce stade, on a un array qui contient N_frames frames de mesures avec les chirps a la suite l'un de l'autre
        chirp_index = 0
        final_array = np.zeros(((Nc - 1) * (N_frame), Ns), dtype=complex)
        for k in range(N_frame):
            for i in range(0, Nc * Ns - Ns, Nc):

                point_index = 0
                for j in range(i,i+Ns,1):
                    final_array[chirp_index,point_index] = full_signal[k][j]
                    point_index+=1
                chirp_index += 1
        #J'ai réarrangé l'array pour que final_array contienne un chirp par ligne. Je n'ai pas séparé les différents frames, on a qu'un seul tableau donc puisqu'un graphe se construit
        #Sur une seule frame, lorsque je plot je prend Nc lignes du tableau par graphe. 1 frame s'étend sur Nc lignes.

        #Réalise et plot les transformées de fourier
        if(graph):
            freq_y = np.fft.fftfreq(Ns, Ts)

            freq_y = c*Ns/(2*B)

            freq_x = np.fft.fftfreq(Nc, Ns * Ts)
            freq_x = np.fft.fftshift(freq_x)
            v_max = 10
        #mais je pense que c'est plutot une moyenne sur les lignes qu'il faut faire pour éliminer la composante DC
            for i in range(0, len(final_array), Nc):
                mes1 = final_array[i:i + Nc]
                for j in range(len(mes1[0])):
                    mes1[:,j] = mes1[:,j] - np.mean(mes1[:,j])
    #La composante DC est supprimée, on peut maintenant réaliser la FFT et la tourner de 90 degrés
                fft_final = np.rot90(np.fft.fftshift(np.fft.fft2(mes1),axes=(0,)))

                print("Calcul en cours du frame #" + str(i/Nc +1)+ " du fichier : "+name)
                #La bah on plot juste en vrai
                plt.imshow(abs(fft_final), extent=[-v_max, v_max, 0, freq_y], aspect='auto')
                plt.xlabel('v (m/s)');plt.ylabel('d (m)')
                plt.title(f'Frame {i/Nc +1} of {N_frame}')
                if(platform.system() == 'Windows'):
                    save_path = '%s\\fft\\%s\\fft_%i.jpg' %(source_path,base_name[0:-4],i/Nc +1)
                else:
                    save_path = '%s/fft/%s/fft_%i.jpg' % (source_path, base_name[0:-4], i / Nc + 1)
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.colorbar()

                plt.savefig(save_path, dpi=100)
                plt.clf()




#Récupère le nom du répertoire courant. Les fichiers d'entrées doivent s'y trouver dans le dossier "mesures"
#Les fichiers de sorties porteront le même nom mais dans le dossier "fft".
"""
source_path = os.getcwd()
save_path = os.getcwd()
if(platform.system() == 'Windows'):
    files = os.listdir("%s\\mesures" % source_path)
else:
    files = os.listdir("%s/mesures"%source_path)
for file in files:
    if file != ".DS_Store":
        print(file)
        if (platform.system() == 'Windows'):
            file2 = "%s\\mesures\\%s" % (source_path, file) #Windows
        else:
            file2 = "%s/mesures/%s" %(source_path,file) #Mac et Linux
        fem(file2,file,source_path,graph = True,only_load=False)
"""

angle()