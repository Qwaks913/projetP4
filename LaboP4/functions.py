import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import platform
from cmath import polar, exp, phase, rect
from math import radians



def calib_angle():
    file = 'GR13_mesure_8m_en_face.npz'
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


def fem(name, base_name=None, source_path=None, only_load = 0,calibration = True,verbose = False):
    # name = 'GR13_MES1_11m.npz'
    with np.load(name, allow_pickle=True) as mes:
        # On importe les données (si les colonnes existent)
        data = mes['data']
        if 'data_times' in mes:
            data_times = mes['data_times']
        background = mes['background']
        if 'background_times' in mes:
            background_times = mes['background_times']
        chirp = mes['chirp']
        f0 = chirp[0]  # Fréquence de la porteuse
        B = chirp[1]  #
        Ns = int(chirp[2])  # Le nombre de points par chirp (utiles uniquement)
        Nc = int(chirp[3])  # Le nombre de chirp dans une frame
        Ts = (chirp[4])  # Période d'échantillonage
        Tc = (chirp[5])  # Période de répétition des chirps en seconde
        if 'datetime' in mes:
            datetime = mes['datetime']

        N_frame = len(data)
        c = 3 * 10 ** 8
        # On print les données
        print("B = " + str(B))
        print("Ns = " + str(Ns))
        print("Fs = " + str(1 / Ts))
        print("f0 = " + str(f0))
        print("Tc = " + str(Tc))
        print("Nc = " + str(Nc))
        print("N_frames = " + str(N_frame))

        to_Throw = int(len(data[0][0]) - Ns * Nc)  # nombre total de points à supprimer par frame
        to_Throw_chirp = int(to_Throw / Nc)  # Nombre de points à supprimer par chirp
        newLen = int(len(data[0][0]) - to_Throw)  # définit la taille d'un frame après avoir retiré les points de pause
        usefull = 1 - to_Throw_chirp / Ns
        I_1 = np.zeros((len(data), newLen))
        Q_1 = np.zeros((len(data), newLen))
        I_2 = np.zeros((len(data), newLen))
        Q_2 = np.zeros((len(data), newLen))

        # On supprime les points de pauses
        for i in range(len(data)):
            indexes = []
            for j in range(Ns, (Ns + to_Throw_chirp) * Nc, Ns + to_Throw_chirp):
                for k in range(j, j + to_Throw_chirp):
                    indexes.append(k)
            I_1[i] = np.delete(data[i][0], indexes)
            I_2[i] = np.delete(data[i][2], indexes)
            Q_1[i] = np.delete(data[i][1], indexes)
            Q_2[i] = np.delete(data[i][3], indexes)


        # l'array data contient N_frames frames contenant chacuns les points recues par les différentes antennes
        if (only_load == 2):
            return (I_1, I_2, Q_1, Q_2, Ns)

        # Nous disposons maintenant des données recueillies par l'antenne sans les pauses.
        full_signal1 = I_1 + complex(0, -1) * Q_1  # on additionne les parties réelles et immaginaires
        full_signal2 = I_2 + complex(0, -1) * Q_2  # on additionne les parties réelles et immaginaires
        if(only_load != 3 and calibration == True):
            phi = calib_angle()
            full_signal2 = full_signal2*exp(+1j*phi)
        # A ce stade, on a un array qui contient N_frames frames de mesures avec les chirps a la suite l'un de l'autre
        chirp_index = 0
        final_array1 = np.zeros(((Nc) * (N_frame), Ns), dtype=complex)
        final_array2 = np.zeros(((Nc) * (N_frame), Ns), dtype=complex)
        #Pour chaque frame
        for k in range(N_frame):
            '''print(" k = " + str(k))
            print("N_frame = " + str(N_frame))
            print(" FileName = " + str(name))
            print()'''
            t = data_times[k]
            #i prend l'indice du point qui marque le début du chirp

            #Nc nombre de p-chirp par frame
            #Ns nombre de points par chirp
            for i in range(0, Nc * Ns - Ns, Ns):

                point_index = 0
                for j in range(i, i + Ns, 1):
                    '''print("i = "+str(i))
                    print(" j = "+ str(j))

                    print(" chirp index = " + str(chirp_index))
                    print(" point_index = " + str(point_index))'''
                    final_array1[chirp_index, point_index] = full_signal1[k][j]
                    final_array2[chirp_index, point_index] = full_signal2[k][j]
                    point_index += 1
                chirp_index += 1
        # J'ai réarrangé l'array pour que final_array contienne un chirp par ligne. Je n'ai pas séparé les différents frames, on a qu'un seul tableau donc puisqu'un graphe se construit
        # Sur une seule frame, lorsque je plot je prend Nc lignes du tableau par graphe. 1 frame s'étend sur Nc lignes.

        # Réalise et plot les transformées de fourier



        d_max = c * Ns / (2 * B)


        v_max = int(c/(4*f0*Tc*2))


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
            """fft1 = np.fft.fftshift(np.fft.fft2(mes1), axes=(0,))
            fft2 = np.fft.fftshift(np.fft.fft2(mes2), axes=(0,))"""
            fft1 = np.fft.fftshift(np.fft.fft2(mes1,s = (100,100)), axes=(0,))
            fft2 = np.fft.fftshift(np.fft.fft2(mes2,s = (100,100)), axes=(0,))

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
                facteur_correcteur_dist = -4.57
                distance = (len(fft_final) - max_indexes[0]) / len(fft_final) * d_max + facteur_correcteur_dist
                if(verbose):
                    print("longueur de la fft " + str(fft_final.shape))
                    print("Calcul en cours du frame #" + str(i / Nc + 1) + " du fichier : " + name)
                    print("max indexes = "+str(max_indexes))
                    print("len = " + str(fft_final.shape))
                    print("distance = " + str(distance))
                    print("angle = " + str(angle))
                x = distance*np.sin(radians(angle))
                y = abs(distance * np.cos(radians(angle)))

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
                fig.suptitle( "Angle de la cible principale: " + str(round(angle, 2)) + "°")

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
                plt.close(fig)
            index_frame+=1
        print("Traitement du signal terminé")
        return array_of_frames1,array_of_frames2,array_of_maxs_indexes



# Récupère le nom du répertoire courant. Les fichiers d'entrées doivent s'y trouver dans le dossier "mesures"
# Les fichiers de sorties porteront le même nom mais dans le dossier "fft".