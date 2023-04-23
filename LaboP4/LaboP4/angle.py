import numpy as np
#l'antenne de droite a les entrées I_1 et Q_1 
from cmath import polar, exp, phase, rect
#name = 'GR13_MES1_11m.npz'
with np.load("C:/Users/bentr/Documents/UCL/Q6/LEPL1508 - Projet 4 en electricité/Labo 2/mesures/GR13_mesure_4m_en face.npz", allow_pickle=True) as mes:
    print(mes.files)
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
    #print(polar((data[0][0]+complex (0,-1)*data[0][2])[2]))
    #print(polar((data[0][1]+complex (0,-1)*data[0][3])[2]))

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

    sig_1 = I_1+complex(0,-1)*Q_1
    sig_2 = I_2+complex(0,-1)*Q_2
    
    angles_0 = np.zeros(len(sig_1))
    for i in range (len(sig_1)) :
        angles_0[i] = phase(np.dot(sig_1[i], np.conjugate(sig_2[i]))/(np.linalg.norm(sig_1[i])*np.linalg.norm(sig_2[i])))
        
    phi_0 = np.mean(angles_0)
    print(phi_0)


with np.load("C:/Users/bentr/Documents/UCL/Q6/LEPL1508 - Projet 4 en electricité/Labo 2/mesures/GR13_4m_2l.npz", allow_pickle=True) as mes:
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
    lambd = c/f0
    print(lambd)
    k_=2*np.pi/lambd
    #
    #print(polar((data[0][0]+complex (0,-1)*data[0][2])[2]))
    #print(polar((data[0][1]+complex (0,-1)*data[0][3])[2]))

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
    
    print(k_)
    sg_a1 = I_1+complex(0,-1)*Q_1
    sg_a2 = I_2+complex(0,-1)*Q_2*exp(1j*phi_0)



    angles = np.zeros(len(sg_a1))
    for i in range(len(sg_a1)) :
            angles[i] = phase(np.dot(sg_a1[i], sg_a2[i]))/(np.linalg.norm(sg_a1[i])*np.linalg.norm(sg_a2[i]))

    print(angles)
    print(np.degrees(np.mean(angles))-90)
   