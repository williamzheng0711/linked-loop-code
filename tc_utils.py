import numpy as np


def Tree_symbols_to_bits(listSize, L, J, sigPos):
    cs_decoded_tx_message = -1* np.ones((listSize, L*J))
    for id_row in range(listSize):
        for id_col in range(L):
            if sigPos[id_row, id_col] != -1:
                a = np.binary_repr(sigPos[id_row, id_col], width=J)      # print("a = " + str(a))
                b = np.array([int(n) for n in a] ).reshape(1,-1)         # print("b = " + str(b))
                cs_decoded_tx_message[id_row, id_col*J:(id_col+1)*J] = b[0,:]
    return cs_decoded_tx_message


def Tree_encode(tx_message,K,G,L,J,P,Ml,messageLengthVector,parityLengthVector):
    encoded_tx_message = np.zeros((K,Ml+P),dtype=int)
    encoded_tx_message[:,0:messageLengthVector[0]] = tx_message[:,0:messageLengthVector[0]]
    for i in range(1,L):
        ParityInteger=np.zeros((K,1),dtype='int')
        G1=G[i-1]
        for j in range(1,i+1):
            ParityBinary = np.mod(np.matmul(tx_message[:,np.sum(messageLengthVector[0:j-1]):np.sum(messageLengthVector[0:j])],
                                G1[np.sum(messageLengthVector[0:j-1]):np.sum(messageLengthVector[0:j])]),2)
            # Convert into decimal equivalent\n",
            ParityInteger1 = ParityBinary.dot(2**np.arange(ParityBinary.shape[1])[::-1]).reshape([K,1])
            ParityInteger = np.mod(ParityInteger+ParityInteger1,2**parityLengthVector[i])
        # Convert integer parity back into bit    \n",
        Parity = np.array([list(np.binary_repr(int(x),parityLengthVector[i])) for x in ParityInteger], dtype=int)
        encoded_tx_message[:,i*J:i*J+messageLengthVector[i]] = tx_message[:,np.sum(messageLengthVector[0:i]):np.sum(messageLengthVector[0:i+1])]
        # Embed Parity check bits\n",
        encoded_tx_message[:,i*J+messageLengthVector[i]:(i+1)*J] = Parity
    
    return encoded_tx_message


def generate_parity_matrix(L,messageLengthVector,parityLengthVector):
    # Generate a full matrix, use only the portion needed for tree code
    G = []
    for i in range(1,L):
        Gp = np.random.randint(2,size=(np.sum(messageLengthVector[0:i]),parityLengthVector[i])).tolist()
        G.append(Gp)
    return np.asarray(G, dtype=object)


def Tree_decoder(cs_decoded_tx_message,G,L,J,B,parityLengthVector,messageLengthvector,listSize):
    tree_decoded_tx_message = np.empty(shape=(0,0))
    for i in range(listSize):
        Paths = np.array([[i]])
        for l in range(1,L):
            # Grab the parity generator matrix corresponding to this section
            G1 = G[l-1]
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # Compute the permissible parity check bits for the section
                Parity_computed = compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector)
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 
        if Paths.shape[0] >= 2:
            # If tree decoder outputs multiple paths for a root node, select the first one 
            flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            if flag:
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            else:
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
                # tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
        elif Paths.shape[0] == 1:
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    return tree_decoded_tx_message


def check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector):   
    msg_bits = extract_msg_bits(Paths,cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    flag = (msg_bits == msg_bits[0]).all()    
    return flag


def compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthVector):
    msg_bits = extract_msg_bits(Path,cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
    Lpath = Path.shape[1]
    Parity_computed_integer = 0
    for i in range(Lpath):
        ParityBinary = np.mod(np.matmul(msg_bits[:,np.sum(messageLengthVector[0:i]):np.sum(messageLengthVector[0:i+1])],
                            G1[np.sum(messageLengthVector[0:i]):np.sum(messageLengthVector[0:i+1])]),2)
        ParityBinary=ParityBinary.reshape(1,-1)
        # Convert into decimal equivalent\n",
        ParityInteger1 = ParityBinary.dot(2**np.arange(ParityBinary.shape[1])[::-1])
        Parity_computed_integer = np.mod(Parity_computed_integer+ParityInteger1,2**parityLengthVector[Lpath])        
         
    Parity_computed = np.array([list(np.binary_repr(int(x),parityLengthVector[Lpath])) for x in Parity_computed_integer], dtype=int)
    return Parity_computed


def parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector):
    index=0
    Lpath = Path.shape[1]
    Parity = cs_decoded_tx_message[k,Lpath*J+messageLengthvector[Lpath]:(Lpath+1)*J]
    if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
        index = 1
    
    return index

def extract_msg_bits(Paths,cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector):
    msg_bits = np.empty(shape=(0,0))
    L1 = Paths.shape[0]
    for i in range(L1):
        msg_bit=np.empty(shape=(0,0))
        path = Paths[i].reshape(1,-1)
        for j in range(path.shape[1]):
            msg_bit = np.hstack((msg_bit,cs_decoded_tx_message[path[0,j],J*j:J*j+messageLengthVector[j]].reshape(1,-1))) if msg_bit.size else cs_decoded_tx_message[path[0,j],J*(j):J*(j)+messageLengthVector[j]]
            msg_bit=msg_bit.reshape(1,-1)
        msg_bits = np.vstack((msg_bits,msg_bit)) if msg_bits.size else msg_bit           
    return msg_bits