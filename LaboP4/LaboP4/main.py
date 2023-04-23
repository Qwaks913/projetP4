import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os


def mes():
    print("data : ")
    print(data)
    print("data_times : ")
    print(data_times)
    print("background : ")
    print(background)
    print("background_times : ")
    print(background_times)
    print("chirp : ")
    print(chirp)
    print("datetime : ")
    print(datetime)

def fem(name,base_name,source_path):

    #name = 'GR13_MES1_11m.npz'
    with np.load(name, allow_pickle=True) as mes:
        data = mes['data']
        #data_times = mes['data_times']
        background = mes['background']
        #background_times = mes['background_times']
        chirp = mes['chirp']
        f0 = chirp[0] #Fréquence de la porteuse
        B = chirp[1] #
        Ns = int(chirp[2]) #Le nombre de points par chirp (utiles uniquement)
        Nc = int(chirp[3]) #Le nombre de chirp dans une frame
        Ts = (chirp[4]) #Période d'échantillonage
        Tc = (chirp[5]) #Période de répétition des chirps en seconde
        N_frame = len(data)
        c = 3*10**8
        #

        to_Throw = int(len(data[0][0]) - Ns * Nc) #nombre total de points à supprimer par frame
        to_Throw_chirp = int(to_Throw / Nc) #Nombre de points à supprimer par chirp
        newLen = int(len(data[0][0])-to_Throw) #définit la taille d'un frame après avoir retiré les points de pause
        I_1 = np.zeros((len(data), newLen))
        Q_1 = np.zeros((len(data), newLen))
        I_2 = np.zeros((len(data), newLen))
        Q_2 = np.zeros((len(data), newLen))
        final_array = np.zeros((Nc * N_frame, Ns))



        for i in range(len(data)):
            indexes = []
            for j in range (Ns, (Ns+to_Throw_chirp) * Nc, Ns + to_Throw_chirp):
                for k in range(j, j + to_Throw_chirp):
                    indexes.append(k)
            I_1[i] = np.delete(data[i][0],indexes)
            I_2[i] = np.delete(data[i][2],indexes)
            Q_1[i] = np.delete(data[i][1],indexes)
            Q_2[i] = np.delete(data[i][3],indexes)
        datetime = mes['datetime']
        #l'array data contient 20 frames contenant chacuns les points recues par les différentes antennes

        #Nous disposons maintenant des données recueillies par l'antenne sans les pauses.
        full_signal = I_1+complex(0,-1)*Q_1 #on additionne les parties réelles et immaginaires
        #print(full_signal)
        #A ce stade, on a un array qui contient 20 frames de mesures avec les chirps a la suite l'un de l'autre
        chirp_index = 0
        final_array = np.zeros(((Nc - 1) * (N_frame), Ns), dtype=complex)
        for k in range(N_frame):
            for i in range(0, Nc * Ns - Ns, Nc):

                point_index = 0
                for j in range(i,i+Ns,1):
                    final_array[chirp_index,point_index] = full_signal[k][j]
                    point_index+=1
                chirp_index += 1
        #J'ai réarrangé l'array pour que final_array contienne un chirp par ligne. attention, j'ai mis tous les frames a la suite et je ne suis pas sur
        #Qu'il faille le faire comme cela puisque ça nous donnerait qu'une seule image pour plusieurs frames ce qui est étrange.
        #Je sépare l'array en différents frames par la suite pour afficher les différentes images


    #Pour faire une moyenne sur les colonnes
        '''for i in range(0,len(final_array),Mc):
            mes1 = final_array[i:i+Mc]
            for j in range(len(mes1)):
                mes1[j] = mes1[j] - np.mean(mes1[j])
            """plt.plot(mes1[0])
            plt.show()"""
            mes1[0][0] = mes1[0][1]
            fft_final = np.log(np.fft.fft2(mes1))
            sb.heatmap(np.abs(fft_final))
            plt.show()'''
        freq_y = np.fft.fftfreq(Ns, Ts)
        freq_y = np.fft.fftshift(freq_y)

        freq_y = c * freq_y / 2 / B * Ns * Ts

        freq_x = np.fft.fftfreq(Nc, Ns * Ts)
        freq_x = np.fft.fftshift(freq_x)
        v_max = 10
    #mais je pense que c'est plutot une moyenne sur les lignes
        for i in range(0, len(final_array), Nc):
            mes1 = final_array[i:i + Nc]
            for j in range(len(mes1[0])):
                mes1[:,j] = mes1[:,j] - np.mean(mes1[:,j])

            fft_final = np.rot90(np.fft.fftshift(np.fft.fft2(mes1)))

            print("Calcul en cours du frame #" + str(i/Nc +1)+ " du fichier : "+name)

            freq_x = (freq_x - f0) * c / 2 * (Nc * Tc)
            plt.imshow(abs(fft_final), extent=[-v_max, v_max, freq_y[0], freq_y[-1]], aspect='auto')
            plt.xlabel('v (m/s)');plt.ylabel('d (m)')
            plt.title(f'Frame {i/Nc +1} of {N_frame}')
            save_path = '%s\\fft\%s\\fft_%i.pdf' %(source_path,base_name[0:-4],i/Nc +1)
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.colorbar()

            plt.savefig(save_path, dpi=100)
            plt.clf()





source_path = 'C:\\Users\\Augustin\\Desktop\\LaboP4'
save_path = 'C:\\Users\\Augustin\\Desktop\\LaboP4'
files = os.listdir("%s\\mesures"%source_path)
for file in files:
    if file != ".DS_Store":
        print(file)
        file2 = "%s\\mesures\\%s" %(source_path,file)
        fem(file2,file,source_path)
