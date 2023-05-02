import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import platform
from cmath import polar, exp, phase, rect

def calib_distance(B,Ts,spectrogramme_2d_shape,pos_cible):
    c = 299792458  # Vitesse de la lumière en m/s
    radar_data = np.load('radar_data.npy')
    distance_cible = 4
    temps_de_vol = pos_cible[1]  # Temps de vol en échantillons
    freq_cible = (pos_cible[0] / radar_data.shape[0]) * Ts  # Fréquence en Hz
    distance_calibree = (freq_cible * temps_de_vol * c) / (2 * B)
    # Convertir les positions en distance
    distance_array = np.zeros(spectrogramme_2d_shape)
    for i in range(spectrogramme_2d_shape[1]):
        distance_array[:, i] = distance_calibree + (i / Ts) * (c / 2)
    return distance_array

"""def calib_angle():
    file = 'calib0.npz'
    source_path = os.path.abspath(".")
    if (platform.system() == 'Windows'):
        name_cal = "%s\\LaboP4\\calibration_file\\%s" % (source_path, file)  # Windows
    else:
        name_cal = "%s/calibration_file/%s" % (source_path, file)  # Mac et Linux
    fem(name_cal,only_load = )"""


def angle2():

    calibration = True
    source_path = os.path.abspath(".")
    c = 3 * 10 ** 8
    lam = c / (2.4 * 10 ** 9)
    k = 2 * np.pi / lam
    file = 'calib0.npz'
    if (platform.system() == 'Windows'):
        name_cal = "%s\\LaboP4\\calibration_file\\%s" % (source_path, file)  # Windows
    else:
        name_cal = "%s/calibration_file/%s" % (source_path, file)  # Mac et Linux
    # mesure
    file = 'GR13_4m_4l.npz'
    if (platform.system() == 'Windows'):
        name_file = "%s\\LaboP4\\mesures\\%s" % (source_path, file)  # Windows
    else:
        name_file = "%s/mesures/%s" % (source_path, file)  # Mac et Linux
    I1_cal, Q1_cal, I2_cal, Q2_cal, Ns_cal = fem(name_cal,only_load=2)
    mes1, mes2, max = fem(name_file,only_load=1)

    cal1 = I1_cal + complex(0, -1) * Q1_cal
    cal2 = I2_cal + complex(0, -1) * Q2_cal
    if (calibration):

        phis = np.linspace(0, 2 * np.pi, num=300)
        maxNorm = 0
        phi_0 = 0
        for phi in phis:
            tempoNorm = np.linalg.norm(np.dot(cal1.flatten(), cal2.flatten() * np.exp(-1j * phi)))
            if (tempoNorm > maxNorm):
                maxNorm = tempoNorm
                phi_0 = phi
        print("phi0 = ")
        print(np.degrees(phi_0))

        # alpha = phase(np.mean(cal1))-phase(np.mean(cal2))
        # différence de phase sur tous les points et faire la moyenne de cette différence
        print("phase de mes2 sans calibration" + str(phase(mes2[max])))
        mes2 = mes2 * exp(-1j * phi_0)
        print("phase de mes2 avec calibration" + str(phase(mes2[max])))
    deph = phase(mes1[max])-phase(mes2[max])
    angle = 90-np.degrees(np.arccos(deph / np.pi))
    print("angle = "+str(angle))



def angle(measure_file,calib_file = 'calib0.npz'):
    calibration = True
    delete_dc = True
    source_path = os.path.abspath(".")
    c = 3 * 10 ** 8
    lam = c / (2.4 * 10 ** 9)
    k = 2 * np.pi / lam
    if (platform.system() == 'Windows'):
        name_cal = "%s\\LaboP4\\calibration_file\\%s" % (source_path, calib_file)  # Windows
    else:
        name_cal = "%s/calibration_file/%s" % (source_path, calib_file)  # Mac et Linux
    # mesure
    if (platform.system() == 'Windows'):
        name_file = "%s\\LaboP4\\mesures\\%s" % (source_path, measure_file)  # Windows
    else:
        name_file = "%s/mesures/%s" % (source_path, measure_file)  # Mac et Linux
    I1_cal, Q1_cal, I2_cal, Q2_cal, Ns_cal = fem(name_cal,only_load=2)
    I1_mes, Q1_mes, I2_mes, Q2_mes, Ns_mes = fem(name_file,only_load=2)
    # Reconstitution des signaux

    cal1 = I1_cal + complex(0, -1) * Q1_cal
    if (delete_dc):
        for i in range(len(cal1)):
            cal1[i:i + Ns_cal] = cal1[i:i + Ns_cal] - np.mean(cal1[i:i + Ns_cal])

    cal2 = I2_cal + complex(0, -1) * Q2_cal
    if (delete_dc):
        for i in range(len(cal1)):
            cal2[i:i + Ns_cal] = cal2[i:i + Ns_cal] - np.mean(cal2[i:i + Ns_cal])
    mes1 = I1_mes + complex(0, -1) * Q1_mes
    if (delete_dc):
        for i in range(len(mes1)):
            mes1[i:i + Ns_mes] = mes1[i:i + Ns_mes] - np.mean(mes1[i:i + Ns_mes])
    mes2 = I2_mes + complex(0, -1) * Q2_mes
    if (delete_dc):
        for i in range(len(cal1)):
            mes2[i:i + Ns_mes] = mes2[i:i + Ns_mes] - np.mean(mes2[i:i + Ns_mes])
    angles_0 = np.zeros(len(cal1))
    if (calibration):

        phis = np.linspace(0, 2 * np.pi, num=300)
        maxNorm = 0
        for phi in phis:
            tempoNorm = np.linalg.norm(np.dot(cal1.flatten(), cal2.flatten() * np.exp(-1j * phi)))
            if (tempoNorm > maxNorm):
                maxNorm = tempoNorm
                phi_0 = phi

        print(np.degrees(phi_0))

        # alpha = phase(np.mean(cal1))-phase(np.mean(cal2))
        # différence de phase sur tous les points et faire la moyenne de cette différence
        mes2 = mes2 * exp(-1j * phi_0)
    test_angle = np.linspace(0, np.pi, 360)
    alpha = np.zeros(len(mes1))
    count = 0
    for i in range(len(mes1)):
        max = 0
        for j in range(len(test_angle)):
            res = np.dot(mes1[i], (mes2[i] * exp(-1j * test_angle[j])))
            # print(np.linalg.norm(res), max)
            if (np.linalg.norm(res)) > max:
                alpha[i] = phase(res)
                max = res

    angle = 90-np.degrees(np.arccos(alpha / np.pi))
    print(angle)


def fem(name, base_name=None, source_path=None, only_load = 0):
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
        # A ce stade, on a un array qui contient N_frames frames de mesures avec les chirps a la suite l'un de l'autre
        chirp_index = 0
        #final_array1 = np.zeros(((Nc - 1) * (N_frame), Ns), dtype=complex)
        #final_array2 = np.zeros(((Nc - 1) * (N_frame), Ns), dtype=complex)
        final_array1 = np.zeros(((Nc) * (N_frame), Ns), dtype=complex)
        final_array2 = np.zeros(((Nc) * (N_frame), Ns), dtype=complex)
        for k in range(N_frame):

            t = data_times[k]
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

        freq_y = np.fft.fftfreq(Ns, Ts)

        freq_y = c * Ns / (2 * B)

        freq_x = np.fft.fftfreq(Nc, Ns * Ts)
        freq_x = np.fft.fftshift(freq_x)
        v_max = int(c/(4*f0*Tc*2))
        # mais je pense que c'est plutot une moyenne sur les lignes qu'il faut faire pour éliminer la composante DC
        array_of_frames1 = np.zeros((N_frame,Nc,Ns),dtype=np.complex128)
        array_of_frames2 = np.zeros((N_frame, Nc, Ns),dtype=np.complex128)
        array_of_maxs_indexes = np.zeros((N_frame,2))
        index_frame = 0
        for i in range(0, len(final_array1), Nc):

            mes1 = final_array1[i:i + Nc]
            mes2 = final_array2[i:i + Nc]
            # Ouvrir le fichier en mode append (ajout)
            """debug_file = "debug.txt"
            with open(debug_file, "a") as f:
                f.write("Début du frame à t = "+str(t))
                np.savetxt(f, full_signal1)"""

            for j in range(len(mes1[0])):
                mes1[:, j] = mes1[:, j] - np.mean(mes1[:, j])
                mes2[:, j] = mes2[:, j] - np.mean(mes2[:, j])
            # La composante DC est supprimée, on peut maintenant réaliser la FFT et la tourner de 90 degrés
            fft1 = np.fft.fftshift(np.fft.fft2(mes1), axes=(0,))
            fft2 = np.fft.fftshift(np.fft.fft2(mes2), axes=(0,))
            fft_final = np.rot90(fft1)
            #trouver le max de la FFT:
            max = np.argmax(fft1)
            max_indexes = np.unravel_index(max, fft1.shape)
            array_of_frames1[index_frame] = mes1
            array_of_frames2[index_frame] = mes2
            array_of_maxs_indexes[index_frame] = max_indexes

            if (only_load==0):
                print("longueur de la fft " + str(fft_final.shape))

                print("Calcul en cours du frame #" + str(i / Nc + 1) + " du fichier : " + name)
                # La bah on plot juste en vrai
                plt.imshow(abs(fft_final), extent=[-v_max, v_max, -3, freq_y-3], aspect='auto')
                plt.xlabel('v (m/s)')
                plt.ylabel('d (m)')
                plt.title(f'Frame {i / Nc + 1} of {N_frame}')
                if (platform.system() == 'Windows'):
                    save_path = '%s\\fft\\%s\\fft_%i.jpg' % (source_path, base_name[0:-4], i / Nc + 1)
                else:
                    save_path = '%s/fft/%s/fft_%i.jpg' % (source_path, base_name[0:-4], i / Nc + 1)
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.colorbar()

                plt.savefig(save_path, dpi=100)
                plt.clf()
            index_frame+=1
        return array_of_frames1,array_of_frames2,array_of_maxs_indexes


# Récupère le nom du répertoire courant. Les fichiers d'entrées doivent s'y trouver dans le dossier "mesures"
# Les fichiers de sorties porteront le même nom mais dans le dossier "fft".




#angle()