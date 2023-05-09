import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import platform
from cmath import polar, exp, phase, rect
from math import radians

c = 3 * 10 ** 8



def calib_angle():
    file = 'Calib-fin.npz'
    source_path = os.path.abspath(".")
    if (platform.system() == 'Windows'):
        name_cal = "%s\\LaboP4\\calibration_file\\%s" % (source_path, file)  # Windows
    else:
        name_cal = "%s/calibration_file/%s" % (source_path, file)  # Mac et Linux
    fft1,fft2 = fem(name_cal,only_load = 3)
    max1 = np.argmax(abs(fft1))
    max_indexes1 = np.unravel_index(max1, fft1.shape)
    max2 = np.argmax(abs(fft2))
    max_indexes2 = np.unravel_index(max2, fft2.shape)
    max1_complex = fft1[max_indexes1]/np.abs(fft1[max_indexes1])
    max2_complex = fft2[max_indexes1]/np.abs(fft2[max_indexes1])

    phi = phase(max1_complex/max2_complex)
    return phi

def calc_angle(fft1,fft2):
    c = 3 * 10 ** 8
    lam = c / (2.4 * 10 ** 9)
    k = 2 * np.pi / lam
    d = lam/2
    max1 = np.argmax(fft1)
    max_indexes1 = np.unravel_index(max1, fft1.shape)
    test_angle =np.linspace(0, np.pi, num=360)
    max = 0
    phi = 0
    for j in range(len(test_angle)):

        res = np.abs(fft1[max_indexes1]+fft2[max_indexes1]*exp(-1j*k*d*np.cos(test_angle[j])))

        if (res) > max:
            phi = test_angle[j]
            max = res
    angle = np.degrees(phi)
    return 90-angle
    print("L'angle de la cible principale est de :" + str(90-angle) + "°")

def pr_param(B, Ns, Nc, Ts, Tc, f0, N_frame, d_max, v_max):
    print("\n####### PARAM DE NOS MESURES #######")
    print("B = " + str(B))
    print("Nc = " + str(Nc))
    print("Ns = " + str(Ns))
    print("fs = " + str(1 / Ts))
    print("f0 = " + str(f0))
    print("Ts = " + str(Ts))
    print("Tc = " + str(Tc))
    print("Nb de frames = " + str(N_frame))
    print("d_max = " + str(d_max))
    print("v_max = " + str(v_max) + "\n")

def fem(file_path, base_name=None, source_path=None, only_load = 0,calibration = True, prt = False):
    """
    @Param: *name       = file_path
            *base_name  = file_name
            *source_path= source_path
    """
    with np.load(file_path, allow_pickle=True) as mes:
        # On importe les données (si les colonnes existent)
        unclean_data = mes['data']
        chirp = mes['chirp']

        f0 = chirp[0]               # Fréquence de la porteuse
        B = chirp[1] 
        Ns = int(chirp[2])          # Le nombre de points par chirp (utiles uniquement)
        Nc = int(chirp[3])          # Le nombre de chirp dans une frame
        Ts = (chirp[4])             # Période d'échantillonage
        Tc = (chirp[5])             # Période de répétition des chirps en seconde
        N_frame = len(unclean_data) #Nombres de frames

        d_max = c * Ns / (2 * B)
        v_max = int(c/(4*f0*Tc))

        if(prt):
            pr_param(B, Ns, Nc, Ts, Tc, f0, N_frame, d_max, v_max)

        N = unclean_data.shape[2]
        
        to_Throw = int(N - Ns * Nc)  # nombre total de points à supprimer par frame
        N_tot = int(N/Nc)  # Nombre de points par chirp
        newLen = int(N - to_Throw)  # définit la taille d'un frame après avoir retiré les points de pause

        #On place toutes les données dons un array clean sans les temps de pauses
        data = np.zeros((N_frame, unclean_data.shape[1], newLen))
        for i in range(Nc):
            data[:, :, i*Ns:(i+1)*Ns] = unclean_data[:, :, i*N_tot:((i*N_tot)+Ns)]

        if (only_load == 2):
            return (data[:, 0, :], data[:, 2, :], data[:, 1, :], data[:, 3, :], Ns)
        
        full_signal1 = data[:, 0, :] - 1j*data[:, 1, :]
        full_signal2 = data[:, 2, :] - 1j*data[:, 3, :]

        if(prt):
            print('e_r_1.shape:', full_signal1.shape)

        if(only_load != 3 and calibration == True):
            phi = calib_angle()
            full_signal2 = full_signal2*exp(+1j*phi)
        
        # A ce stade, on a un array qui contient N_frames frames de mesures avec les chirps a la suite l'un de l'autre

        final_array1 = np.zeros((N_frame, Nc, Ns), dtype=complex)
        final_array2 = np.zeros((N_frame, Nc, Ns), dtype=complex)

        for frame in range(N_frame):
            for nc in range(Nc):
                final_array1[frame, nc, :] = full_signal1[frame, nc*Ns:(nc+1)*Ns]
                final_array2[frame, nc, :] = full_signal2[frame, nc*Ns:(nc+1)*Ns]

        return None
    
        for k in range(N_frame):

            for i in range(0, Nc * Ns - Ns, Nc):

                point_index = 0
                for j in range(i, i + Ns, 1):
                    final_array1[chirp_index, point_index] = full_signal1[k][j]
                    final_array2[chirp_index, point_index] = full_signal2[k][j]
                    point_index += 1
                chirp_index += 1
        # J'ai réarrangé l'array pour que final_array contienne un chirp par ligne. Je n'ai pas séparé les différents frames, on a qu'un seul tableau donc puisqu'un graphe se construit
        # Sur une seule frame, lorsque je plot je prend Nc lignes du tableau par graphe. 1 frame s'étend sur Nc lignes.

        # Réalise et plot les transformées de fourier



        


        array_of_frames1 = np.zeros((N_frame,Nc,Ns),dtype=np.complex128)
        array_of_frames2 = np.zeros((N_frame, Nc, Ns),dtype=np.complex128)
        array_of_maxs_indexes = np.zeros((N_frame,2))
        index_frame = 0
        for i in range(0, len(final_array1), Nc):

            mes1 = final_array1[i:i + Nc]
            mes2 = final_array2[i:i + Nc]
            # Ouvrir le fichier en mode append (ajout)


            for j in range(len(mes1[0])):
                mes1[:, j] = mes1[:, j] - np.mean(mes1[:, j])
                mes2[:, j] = mes2[:, j] - np.mean(mes2[:, j])
            # La composante DC est supprimée, on peut maintenant réaliser la FFT et la tourner de 90 degrés
            fft1 = np.fft.fftshift(np.fft.fft2(mes1), axes=(0,))
            fft2 = np.fft.fftshift(np.fft.fft2(mes2), axes=(0,))

            #fft_final = np.rot90(fft1+fft2)
            fft_final = np.rot90(fft1)
            angle = calc_angle(fft1,fft2)
            #trouver le max de la FFT:
            max = np.argmax(abs(fft_final))
            max_indexes = np.unravel_index(max, fft_final.shape)
            array_of_frames1[index_frame] = mes1
            array_of_frames2[index_frame] = mes2
            array_of_maxs_indexes[index_frame] = max_indexes
            #Utile pour retourner la première fft du fichier de calibration
            if(only_load == 3):
                return fft1,fft2

            if (only_load==0):
                """
                print("longueur de la fft " + str(fft_final.shape))

                print("Calcul en cours du frame #" + str(i / Nc + 1) + " du fichier : " + name)
                print("max indexes = "+str(max_indexes))
                print("len = " + str(fft_final.shape))
                """
                facteur_correcteur_dist = -4.57
                distance = (len(fft_final)-max_indexes[0])/len(fft_final) * d_max +facteur_correcteur_dist
                x = distance*np.sin(radians(angle))
                y = abs(distance * np.cos(radians(angle)))
                """
                print("distance = "+str(distance))
                print("angle = "+str(angle))
                """
                # La bah on plot juste en vrai
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(abs(fft_final), extent=[-v_max, v_max, facteur_correcteur_dist, d_max+facteur_correcteur_dist], aspect='auto')
                #plt.imshow(abs(fft_final), extent=[-v_max, v_max, -3, d_max-3], aspect='auto')
                axs[0].set_xlabel('v (m/s)')
                axs[0].set_ylabel('d (m)')
                axs[0].set_title(f'Frame {i / Nc + 1} of {N_frame}')
                #axs[0].set_figtext(0.7, 0.035, "Angle de la cible principale: " + str(round(angle,2)) + "°", ha='center')

                axs[1].scatter(x,y,color ='red')
                # plt.imshow(abs(fft_final), extent=[-v_max, v_max, -3, d_max-3], aspect='auto')
                axs[1].set_xlabel('x (m)')
                axs[1].set_ylabel('y (m)')
                axs[1].set_xlim([-10, 10])
                axs[1].set_ylim([0, d_max])
                axs[1].set_title(f'Frame {i / Nc + 1} of {N_frame}')
                fig.suptitle( "An gle de la cible principale: " + str(round(angle, 2)) + "°")

                if (platform.system() == 'Windows'):
                    save_path = '%s\\fft\\%s\\fft_%i.jpg' % (source_path, base_name[0:-4], i / Nc + 1)
                else:
                    save_path = '%s/fft/%s/fft_%i.jpg' % (source_path, base_name[0:-4], i / Nc + 1)
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                #axs[0].set_colorbar()
                fig.tight_layout()
                fig.savefig(save_path, dpi=100)
                fig.clf()
            index_frame+=1
        return array_of_frames1,array_of_frames2,array_of_maxs_indexes



# Récupère le nom du répertoire courant. Les fichiers d'entrées doivent s'y trouver dans le dossier "mesures"
# Les fichiers de sorties porteront le même nom mais dans le dossier "fft".
