import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import platform
from cmath import polar, exp, phase, rect
from math import radians

if (platform.system() == 'Windows'):
    win = True
c = 3 * 10 ** 8
lam = c / (2.4 * 10 ** 9)
k = 2 * np.pi / lam
d = lam/2

Calib_file = 'Calib-fin.npz'

if (win):
    Calib_path = f"{os.getcwd()}\\LaboP4\\calibration_file\\{Calib_file}"   # Windows
else:
    Calib_path = f"{os.getcwd()}/LaboP4/calibration_file/{Calib_file}"      # Mac et Linux

def calib_angle(source_path):
    fft1, fft2 = fem(Calib_path, only_load=3)
    max1 = np.argmax(abs(fft1))
    max_indexes1 = np.unravel_index(max1, fft1.shape)
    max2 = np.argmax(abs(fft2))
    max_indexes2 = np.unravel_index(max2, fft2.shape)
    max1_complex = fft1[max_indexes1] / np.abs(fft1[max_indexes1])
    max2_complex = fft2[max_indexes2] / np.abs(fft2[max_indexes2])
    phi = phase(max1_complex / max2_complex)
    return phi

def calc_angle(fft1, fft2):
    max1 = np.argmax(fft1)
    max_indexes1 = np.unravel_index(max1, fft1.shape)
    max2 = np.argmax(fft2)
    max_indexes2 = np.unravel_index(max2, fft2.shape)
    test_angles = np.linspace(0, np.pi, num=360)
    max = 0
    phi = 0
    for i in range(len(test_angles)):
        res = np.abs(fft1[max_indexes1] + fft2[max_indexes2] * exp(-1j*k*d*np.cos(test_angles[i])))
        if(res) > max:
            phi = test_angles[i]
            max = res
    angle = np.degrees(phi)
    return 90-angle

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
            phi = calib_angle(source_path)
            full_signal2 = full_signal2*exp(+1j*phi)
        # A ce stade, on a un array qui contient N_frames frames de mesures avec les chirps a la suite l'un de l'autre

        final_array1 = np.zeros((N_frame, Nc, Ns), dtype=complex)
        final_array2 = np.zeros((N_frame, Nc, Ns), dtype=complex)
        # Final array est un grand tableau avec 1 matrice par frame avec Nc lignes qui contiennent Ns colonnes de données

        angles = np.zeros(N_frame)
        for frame in range(0, N_frame):
            print(f'[{round((frame+1)*100/N_frame, 1)} %] \t completés')
            for nc in range(Nc):
                final_array1[frame, nc, :] = full_signal1[frame, nc*Ns:(nc+1)*Ns]
                final_array2[frame, nc, :] = full_signal2[frame, nc*Ns:(nc+1)*Ns]

            mes1 = final_array1[frame]
            mes2 = final_array2[frame]

            #On supprime la composante DC
            for j in range(Ns):
                mes1[:, j] = mes1[:, j] - np.mean(mes1[:, j])
                mes2[:, j] = mes2[:, j] - np.mean(mes2[:, j])
            
            #On fait la fft et on tourne de 90 degrés
            fft1 = np.fft.fftshift(np.fft.fft2(mes1,s = (100,100)), axes=(0,))
            fft2 = np.fft.fftshift(np.fft.fft2(mes2,s = (100,100)), axes=(0,))
            fft_final = np.rot90(fft1)

            #Utile pour retourner la première fft du fichier de calibration
            if(only_load == 3):
                return fft1,fft2

            angles[frame] = calc_angle(fft1, fft2)
            
            #trouver le max de la FFT:
            max = np.argmax(abs(fft_final))
            max_indexes = np.unravel_index(max, fft_final.shape)

            if (only_load==0):
                facteur_correcteur_dist = -4.57
                distance = (len(fft_final)-max_indexes[0])/len(fft_final) * d_max + facteur_correcteur_dist
                x = distance*np.sin(radians(angles[frame]))
                y = abs(distance * np.cos(radians(angles[frame])))

                # La bah on plot juste en vrai
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(abs(fft_final), extent=[-v_max, v_max, facteur_correcteur_dist, d_max+facteur_correcteur_dist], aspect='auto')
                axs[0].set_xlabel('v (m/s)')
                axs[0].set_ylabel('d (m)')
                axs[0].set_title(f'Frame {frame+1} of {N_frame}')

                axs[1].scatter(x,y,color ='red')
                axs[1].set_xlabel('x (m)')
                axs[1].set_ylabel('y (m)')
                axs[1].set_xlim([-10, 10])
                axs[1].set_ylim([0, d_max])
                axs[1].set_title(f'Frame {frame+1} of {N_frame}')
                fig.suptitle( "Angle de la cible principale: " + str(round(angles[frame], 2)) + "°")

                if (win):
                    save_path = '%s\\fft\\%s\\fft_%i.jpg' % (source_path, base_name[0:-4], frame+1)
                else:
                    save_path = '%s/fft/%s/fft_%i.jpg' % (source_path, base_name[0:-4], frame+1)
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                fig.tight_layout()
                fig.savefig(save_path, dpi=300)
                fig.clf()

        return None

