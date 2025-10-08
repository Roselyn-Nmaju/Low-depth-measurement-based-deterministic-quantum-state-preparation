import numpy as np
import math as math
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate

#insures the input is of legth 2^n
def ModifyVector(vec):
    n = len(vec)
    power2 = 2 ** (n - 1).bit_length() 
    if n < power2:
        vec += [0] * (power2 - n)
    return vec
#function that normalises a list
def NormVector(vec):
    absvals = [i**2 for i in vec]
    s = sum(absvals)
    sqrtsum = np.sqrt(s)
    out = [i/sqrtsum for i in vec]
    vec = out
    return vec

#Generates the rotation angles
def GenAngle(vec,output=None):
    if output is None:
        output = []
    #makes lists that combine 2 items in the list
    if len(vec)>1:
        newx = [0]*int((len(vec))/2)
        for k in range(0,len(newx)):
            eq1 =np.sqrt((abs(vec[2*k])**2) + (abs(vec[2*k+1])**2))
            newx[k]= eq1
        #Calls the function recusively to keep combining elements of the list
        GenAngle(newx,output)
        #Creates the angles from these combined elements and append them to the output
        angles = [0]*int((len(vec))/2)
        for k in range(0,len(newx)):
            eq2 = (vec[2*k+1])/(newx[k])
            if newx[k] != 0:
                if newx[k] > 0:
                    angles[k]=2*np.arcsin(eq2)
                else:
                    angles[k]=2*math.pi - 2*np.arcsin(eq2)
            else:
                angles[k]=0
        output += angles
        return output
    
#Creates the stucture to make a binary tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
#Builds a binary tree 
def BuildTree(vec):
    def CreateNode(index):
        if index >= len(vec) or vec[index] is None:
            return None

        node = TreeNode(vec[index])
        node.left = CreateNode(2 * index + 1)
        node.right = CreateNode(2 * index + 2)
        return node
    
    return CreateNode(0)


#PreorderTraverses a binary tree
def PreorderTraversal(tree):
    result = []

    def Traverse(node):
        if node:
            result.append(node.val)  
            Traverse(node.left)      
            Traverse(node.right)     

    Traverse(tree)
    return result

## Creating the sub-circuits (TE)

#fucntion makes binary strings
def GenBinaryStrings(length):
    binary_strings = []
    max_num = 2 ** length

    for i in range(max_num):
        binary_string = format(i, 'b').zfill(length)
        binary_strings.append(binary_string)
    return binary_strings


#function that reverses a string
def ReverseString(stringlist):
    reversed_list = [string[::-1] for string in stringlist]
    return reversed_list
#produces the TE circuit
def TimeEncoding(rotations):
    index=1
    num_qubits=len(rotations).bit_length()
    #builds blank circuit
    circuit = QuantumCircuit(num_qubits)
    #first rotation is the first item in rotations so can just be directly placed
    circuit.ry(rotations[0],0)

    #generates binaries
    for i in range(1,num_qubits):
        binaries = GenBinaryStrings(i)
        binaries = ReverseString(binaries)
        qbits = list(range(0,i+1))
        #use binaries to construct the controlled y rotations
        for string in binaries:
            control_y = RYGate(rotations[index]).control(i,None,string)
            circuit.append(control_y,qbits)
            index += 1
    return circuit

#Turn binary tree list into binary tree list of sub lists
def TreeSubLists(treelist):
    result = []
    n = len(treelist)
    power = 0
    i = 0
    while i < n:
        sublist = []
        sublist_size = 2 ** power
        for _ in range(sublist_size):
            if i < n:
                sublist.append(treelist[i])
                i += 1
        result.append(sublist)
        power += 1
    return result


#splits a list into splits lists
def SplitList(index_list, splits):
    if splits <= 0 or splits > len(index_list):
        return None  

    sublist_size = len(index_list) // splits
    remainder = len(index_list) % splits

    result = []
    start = 0

    for i in range(splits):
        sublist_end = start + sublist_size + (1 if i < remainder else 0)
        result.append(index_list[start:sublist_end])
        start = sublist_end

    return result



#Groups sub circuit indices
def SubCircuitList(vec,tree_sublist):
    tree_sublist-=1
    split_size=len(vec[tree_sublist])
    output = [[] for _ in range(split_size)]
    
    for sublist in range(tree_sublist,-1,-1):
        for i in range(split_size-1,-1,-1):
            split_list = SplitList(vec[sublist],split_size)
            #appends indexs on the binary tree level to the correct list of indexs that make its sub circuit
            output[i]+=split_list[i]

    return output


## Creating the Circuit (SE)


def QubitGrouper(vec,indices):
    result = []
    n=0
    for i in range(len(indices)):
        sublist = []
        sublist_size = indices[i][1]
        for m in range(sublist_size):
            sublist.append(vec[n+m])
        result.append(sublist)
        n += sublist_size

    return result
#Combining TE and SE
def DAC(input_vec,λ):

    #normalise the state and generate the angles
    y_rotations = GenAngle(NormVector(ModifyVector(input_vec)))

    #index rotations
    vec=list(range(len(y_rotations)))
    vec_len = len(vec)
    vec_sublists = TreeSubLists(vec)
    vec_sublists.reverse()


    number_subtree = int((vec_len + 1)/(2**λ))
    number_subtree_qubit = (2**λ) - 1
    
    tree_list = [[i] for i in range(vec_len - (number_subtree * number_subtree_qubit))]
    tree_list += SubCircuitList(vec_sublists,λ)

    #makes dictionary storing information to create the sub-circuit
    tree_dict =  {i:(tree_list[i], int(math.log(len(tree_list[i])+1,2))) for i in range(len(tree_list))}
    tree_indices = PreorderTraversal(BuildTree(list(range(len(tree_dict)))))
    ordered_list = [tree_dict[key] for key in tree_indices]

    num_qubits = sum([tree_dict[i][1] for i in tree_dict])
    #Creates blank circuit
    circuit=QuantumCircuit(num_qubits)
    # circuit=QuantumCircuit(num_qubits,num_qubits)
   
   #Turn dictionary of index to list of rotations and qubits needed for sub circuit
    ordered_list = [([y_rotations[x] for x in tpl[0]], tpl[1]) for tpl in ordered_list]


    #loads in all the sub circuits
    index=0
    for unitary in range(len(ordered_list)):
        sub_circuit = TimeEncoding(ordered_list[unitary][0])
        circuit=circuit.compose(sub_circuit,list(range(index, index  + ordered_list[unitary][1])))
        index+= ordered_list[unitary][1]

    circuit.barrier()

    grouping_list= QubitGrouper(vec,ordered_list)
    ordering  = PreorderTraversal(BuildTree(list(range(len(grouping_list)))))
    traversed_dict =  {ordering[i]:grouping_list[i] for i in range(len(grouping_list))}

    swap_list = [traversed_dict[key] for key in range(len(grouping_list)) ]
    swap_list = TreeSubLists(swap_list)
    swap_list.reverse()

    #SE method
    m = (len(grouping_list)).bit_length()
    for level in range(1,m):
        circuit.barrier()
        for stage in range(1,level+1):
            for k in range(len(swap_list[level])):
                control = swap_list[level][k]
                qubit_1 = swap_list[level-stage][(2**(stage))*k]
                qubit_2 = swap_list[level-stage][(2**(stage))*k + (2**(stage-1))]

                #loops over qubits in sub circuit 
                for n in range(len(qubit_1)):
                    circuit.cswap(control[0],qubit_1[n],qubit_2[n])
    circuit.barrier() 

    return circuit
            
