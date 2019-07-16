import copy
from scipy.ndimage.measurements import label
import numpy as np


def label_3D(array):

    Lx, Ly, Lz =int(array.shape[0]) , int(array.shape[1]) , int(array.shape[2]) #array lengths
    
    dx,dy,dz = 0, 0 , 0 #find 2 shortest lengths
    
    #these 3 sections create an nx4 array of groupings by x,y,z columns
    if Lx<=Ly or Lx<=Lz:
        dx=1  #set info x length is 1 of 2 shortest
        np1=0
        label_mats=np.zeros((Lx,Ly,Lz))
        for i in range(0,Lx):
            labeled, ncomponents = label(array[i,:,:]) #label 2D array
            lab_add = copy.deepcopy(labeled) #create deepcopy
            lab_add[lab_add>0]=1 #anything above 0 is 1
            lab_add=lab_add*np1  #add the number of components summed up to the new groupings
            label_mats[i,:,:]=labeled+lab_add#add the number of components summed up to the new groupings
            np1=np1+ncomponents #move index
            unique=np.unique(label_mats[i,:,:]) #find unique values in array
            unique=unique[1:] #find unique values in 1D array
            for k in unique:
                inter=copy.deepcopy(label_mats[i,:,:])
                inter[inter!=k]=0
                ind=np.transpose(np.nonzero(inter))
                index_col = i*np.ones((len(ind),1))
                grp_num=k*np.ones((len(ind),1))
                pt_list00 = np.hstack((index_col, ind))
                pt_list0 = np.hstack((grp_num, pt_list00))
                if k==1:
                    pt_listx=pt_list0
                if k>1:
                    pt_listx = np.concatenate((pt_listx, pt_list0))
    
    if Ly<=Lx or Ly<=Lz:
        dy=1 #set info y length is 1 of 2 shortest
        np1=0
        label_mats=np.zeros((Lx,Ly,Lz))          
        for i in range(0,Ly):
            labeled, ncomponents = label(array[:,i,:])
            lab_add = copy.deepcopy(labeled)
            lab_add[lab_add>0]=1
            lab_add=lab_add*np1
            label_mats[:,i,:]=labeled+lab_add
            np1=np1+ncomponents
            unique=np.unique(label_mats[:,i,:])
            unique=unique[1:]
            for k in unique:
                inter=copy.deepcopy(label_mats[:,i,:])
                inter[inter!=k]=0
                ind=np.transpose(np.nonzero(inter))
                index_col = i*np.ones((len(ind),1))
                grp_num=k*np.ones((len(ind),1))
                pt_list00 = np.hstack((index_col, ind))
                pt_list0 = np.hstack((grp_num, pt_list00))
                if k==1:
                    pt_listy=pt_list0
                if k>1:
                    pt_listy = np.concatenate((pt_listy, pt_list0))  
    
    if dx==0 or dy==0:
        dz=1 #set info z length is 1 of 2 shortest
        np1=0
        label_mats=np.zeros((Lx,Ly,Lz))
        for i in range(0,Lz):
            labeled, ncomponents = label(array[:,:,i])
            lab_add = copy.deepcopy(labeled)
            lab_add[lab_add>0]=1
            lab_add=lab_add*np1
            label_mats[:,:,i]=labeled+lab_add
            np1=np1+ncomponents
            unique=np.unique(label_mats[:,:,i])
            unique=unique[1:]
            for k in unique:
                inter=copy.deepcopy(label_mats[:,:,i])
                inter[inter!=k]=0
                ind=np.transpose(np.nonzero(inter))
                index_col = i*np.ones((len(ind),1))
                grp_num=k*np.ones((len(ind),1))
                pt_list00 = np.hstack((index_col, ind))
                pt_list0 = np.hstack((grp_num, pt_list00))
                if k==1:
                    pt_listz=pt_list0
                if k>1:
                    pt_listz = np.concatenate((pt_listz, pt_list0))  
    
    def list_org(list1, list2):
        ind_pt_list2=np.lexsort((list2[:,3],list2[:,2],list2[:,1])) #sort columns by 3,2,1
        list2=list2[ind_pt_list2] #rearrange by previous indices so now col 1-3 for list1 and list2 are same
        for nn in range(int(min(list2[:,0])),int(max(list2[:,0])+1)): #sweep in range from min to max of 0-col
            ind2=np.where(list2[:,0]==nn)[0] #indices where 0-col equals nn value in list2
            x_vals=np.unique(list1[ind2[:],:][:,0]) #list groupings in list 1
            if x_vals.size==1:
                ind1=np.where(list1[:,0]==int(x_vals))[0]
                list1[ind1,0]=x_vals #reassign values for list 1 that share same grp in list2
            else:
                for x in x_vals:
                    ind1=np.where(list1[:,0]==int(x))[0]
                    list1[ind1,0]=min(x_vals) #reassign values for list 1 that share same grp in list2
        return list1
        
    if dx==1 and dy==1:
        pt_listy[:,[1,2]]=pt_listy[:,[2,1]] #rearrange columns
        pt_final=list_org(pt_listx,pt_listy)
    
    if dx==1 and dz==1:
        pt_listz[:,[1,2,3]]=pt_listz[:,[2,3,1]] #rearrange columns
        pt_final=list_org(pt_listx,pt_listz)
        
    if dy==1 and dz==1:
        pt_listy[:,[1,2]]=pt_listy[:,[2,1]] #rearrange columns
        pt_listz[:,[1,3]]=pt_listz[:,[3,1]] #rearrange columns
        pt_final=list_org(pt_listy,pt_listz)
        
    Convert=0*pt_final[:,0]
    for count, elem in enumerate(np.unique(pt_final[:,0]), 1):
        Convert[pt_final[:,0]==elem]=count
    pt_final[:,0]=Convert
    Final=np.zeros((Lx,Ly,Lz))
    for xi in range(len(pt_final)):
        Final[int(pt_final[xi,1]),int(pt_final[xi,2]),int(pt_final[xi,3])]=pt_final[xi,0]    
    elem=np.max(Final)
    
    return pt_final, Final, elem


array=np.zeros((15,14,14))
array[1:10,1:4,1:6]=1
array[2,1:4,1:6]=0
array[5,1:4,1:6]=0
array[1:7,2,2]=1
array[8,1:4,1:6]=0
grp_x_y_z , l_m_n_array , num_3D_grps = label_3D(array)

