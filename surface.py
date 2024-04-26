def random_surface_code(L):
    n = L**2 + (L-1)**2
    LL = 2*L-1
    stabilizer_list = []
    # This is to simplify the parallelization later on
    if L%2==0:
        for j in range(0, LL, 2): 
            for i in range(1, LL, 2):
                stabilizer = []
                if j>0:
                    stabilizer.extend([[j-1,i]])
                stabilizer.extend([[j,i-1], [j,i+1]])
                if j<LL-1:
                    stabilizer.extend([[j+1,i]])
                #print(stabilizer)
                stabilizer_list.append(coord_to_label(stabilizer, L, 'X'))
        for i in range(0, LL, 2): 
            for j in range(1, LL, 2):
                stabilizer = []
                if i>0:
                    stabilizer.extend([[j,i-1]])
                stabilizer.extend([[j-1,i], [j+1,i]])
                if i<LL-1:
                    stabilizer.extend([[j,i+1]])
                stabilizer_list.append(coord_to_label(stabilizer, L, 'Z'))
    else:
        for i in range(1, LL, 2): 
            for j in range(0, LL, 2):
                stabilizer = []
                if j>0:
                    stabilizer.extend([[j-1,i]])
                stabilizer.extend([[j,i-1], [j,i+1]])
                if j<LL-1:
                    stabilizer.extend([[j+1,i]])
                #print(stabilizer)
                stabilizer_list.append(coord_to_label(stabilizer, L, 'X'))
        for j in range(1, LL, 2): 
            for i in range(0, LL, 2):
                stabilizer = []
                if i>0:
                    stabilizer.extend([[j,i-1]])
                stabilizer.extend([[j-1,i], [j+1,i]])
                if i<LL-1:
                    stabilizer.extend([[j,i+1]])
                stabilizer_list.append(coord_to_label(stabilizer, L, 'Z'))
            
    # Parallelized order
    stabilizers_X = stabilizer_list[:(n-1)//2]
    stabilizers_Z = stabilizer_list[(n-1)//2:]
    stabilizer_list = stabilizers_X[::2] + stabilizers_X[1::2] + stabilizers_Z[::2] + stabilizers_Z[1::2]
    
    return stabilizer_list

def coord_to_label(stab, L, case):
    n = L**2 + (L-1)**2
    LL = 2*L-1
    label = list('I'*n)
    #print(stab)
    for ind in stab:
        i,j = ind
        pos = LL*i + j
        pos = (pos+1)//2
        label[pos] = case
    return ''.join(['+'] + list(reversed(label)))