import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt

""" 
   General function for creating a .csv file for simulation of a mobile omnidirectional robot with
   a manipulator.
    
"""



#Initializing transformation matrices for the box and end effector

ang=2.0

Tse = np.array([[0, 0, 1, 0],
                [ 0, 1, 0, 0],
                [ -1, 0, 0, 0.5],
                [ 0, 0, 0, 1]])

# Box origin for 'best' and 'overshoot' case
Tsc_in = np.array([[1, 0, 0, 1],
                    [ 0, 1, 0, 0],
                    [ 0, 0, 1, 0.025],
                    [ 0, 0, 0, 1]])



# Final box position for 'best' and 'overshoot' case
Tsc_fin = np.array([[0, 1, 0, 0],
                    [ -1, 0, 0, -1],
                    [ 0, 0, 1, 0.025],
                    [ 0, 0, 0, 1]])

# Box origin for 'newTask' case
# Tsc_in = np.array([[1, 0, 0, 1.0],
#                     [ 0, 1, 0,0.5],
#                     [ 0, 0, 1, 0.025],
#                     [ 0, 0, 0, 1]])


# Final box position for 'newTask' case 
# Tsc_fin = np.array([[0, 1, 0, 0.2],
#                     [ -1, 0, 0, -0.4],
#                     [ 0, 0, 1, 0.025],
#                     [ 0, 0, 0, 1]])


Tce_sdoff =  np.array([[np.cos(ang), 0, np.sin(ang), 0],
                       [ 0, 1, 0, 0],
                       [ -np.sin(ang), 0, np.cos(ang), 0.15],
                       [ 0, 0, 0, 1]])



Tce_grasp =  np.array([[np.cos(ang), 0, np.sin(ang), 0],
                       [ 0, 1, 0, 0],
                       [ -np.sin(ang), 0, np.cos(ang), 0.01],
                       [ 0, 0, 0, 1]])




def generate_traj(Tse_in,Tsc_in,Tsc_fin,Tce_grasp,Tce_sdoff,k):

    """ 
        This function takes in the initial transformation matrices of the end-effector
        and of the box and returns a final trajectory matrix
        
    """

    # Transforms for the {s} frame to the end effector frame {e}

    Tse_sdoff=Tsc_in@Tce_sdoff
    Tse_grasp=Tsc_in@Tce_grasp
    Tse_fin=Tsc_fin@Tce_sdoff
    Tse_grasp_fin=Tsc_fin@Tce_grasp

    total_traj=[]


    # Generating the eight seperate trajectories

    tf=2.5
    N=tf*k/0.01
    method=3
    traj1=mr.ScrewTrajectory(Tse_in,Tse_sdoff,tf,N,method)
    total_traj.append(traj1)

    traj2=mr.ScrewTrajectory(Tse_sdoff,Tse_grasp,tf,N,method)
    total_traj.append(traj2)

    tf=0.625
    N=tf*k/0.01
    traj3=mr.ScrewTrajectory(Tse_grasp,Tse_grasp,tf,N,method)
    total_traj.append(traj3)
 
    tf=2.5
    N=tf*k/0.01
    traj4=mr.ScrewTrajectory(Tse_grasp,Tse_sdoff,tf,N,method)
    total_traj.append(traj4)
  
    traj5=mr.ScrewTrajectory(Tse_sdoff,Tse_fin,tf,N,method)
    total_traj.append(traj5)


    traj6=mr.ScrewTrajectory(Tse_fin,Tse_grasp_fin,tf,N,method)
    total_traj.append(traj6)


    tf=0.625
    N=tf*k/0.01
    traj7=mr.ScrewTrajectory(Tse_grasp_fin,Tse_grasp_fin,tf,N,method)
    total_traj.append(traj7)


    tf=2.5
    N=tf*k/0.01
    traj8=mr.ScrewTrajectory(Tse_grasp_fin,Tse_fin,tf,N,method)
    total_traj.append(traj8)


    traj_mat=[]
    count=0

    # Appending the trajectorires into one final trajectory matrix 
    for traj in total_traj:
        if count>=2 and count<6:
            gripper_val=np.array([1])   
        else:
            gripper_val=np.array([0])   
        count+=1
        for i in traj:
            traj_row=np.concatenate([i[0,0:3],i[1,0:3],i[2,0:3],i[0:3,3],gripper_val])
            traj_mat.append(traj_row)
         

    traj_mat=np.array(traj_mat)

    return traj_mat


#Calling the function to generate the desired trajectory
k=1.0
desired_traj=generate_traj(Tse,Tsc_in,Tsc_fin,Tce_grasp,Tce_sdoff,k)



l=0.47/2
w=0.15
r=0.0475

def make_SE(ref_traj):
    """ 
    This funciton takes in a 1x13 reference trajectory and converts 
    it to a 4x4 SE(3) matrix.

    """
    T_mat = np.array([[ref_traj[0], ref_traj[1], ref_traj[2], ref_traj[9]],
                    [ ref_traj[3], ref_traj[4], ref_traj[5], ref_traj[10]],
                    [ ref_traj[6], ref_traj[7], ref_traj[8], ref_traj[11]],
                    [ 0,0,0, 1]])

    return T_mat


# Initializing the paramters and measurements for the mobile robot and manipulator

l=0.47/2
w=0.15
r=0.0475

Blist = np.array([[0,     0,       0,       0,      0],
                  [0,    -1,      -1,      -1,     0],
                  [1,     0,       0,       0,      1],
                  [0,    -0.5076,  -0.3526, -0.2176,0],
                  [0.033, 0,       0,       0,      0],
                  [0,     0,       0,       0,      0]])

F=np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
            [ 1, 1, 1, 1],
            [ -1,1,-1, 1]])

F_zeros=np.zeros((1,4))
F=np.vstack((F_zeros,F_zeros,F,F_zeros))

Tb0= np.array([[ 1, 0, 0,0.1662],
               [ 0, 1, 0, 0],
               [ 0, 0, 1, 0.0026],
               [ 0, 0, 0, 1]])

M0e= np.array([[ 1, 0, 0,0.033],
               [ 0, 1, 0, 0],
               [ 0, 0, 1, 0.6546],
               [ 0, 0, 0, 1]])



def FeedbackControl(X, Xd, Xd_next,Xerr_tot, Kp,Ki,dt):

    """
    This function takes in the current and desired states 
    of the robot, and returns the corresponding body twist 
    and body twist error necessary to reach that desired state. 
    
    """


    adjoint = mr.Adjoint(mr.TransInv(X)@Xd)

    Xerr = mr.se3ToVec((mr.MatrixLog6(mr.TransInv(X)@Xd)))

 
    Vd_SE3= (1/dt) * mr.MatrixLog6(mr.TransInv(Xd)@Xd_next)

    Vd=mr.se3ToVec(Vd_SE3)

    Xerr_tot=Xerr_tot+Xerr*dt


    V=adjoint@Vd + Kp@(Xerr) + Ki@(Xerr_tot)
  

    return V, Xerr, Xerr_tot

def NextState(curr_config, control_vec, dt, max_angspeed):

    """
    This function takes in the current configuration, the control vector, 
    and the maximum angular speed of the mobile robot. It calculates
    the next configuration state that the robot has to follow so that 
    it can stay on the desired trajectory. 

    curr_config is a 1x13 configuration vector, and control_vec is a 1x9 
    vector of wheel and arm joint speeds that was caluclated using 
    the body twist from the feedback  controller. 
    
    """


    for i,val in enumerate(control_vec):

        if val < -max_angspeed:
            control_vec[i]=-max_angspeed
        if val>max_angspeed:
            control_vec[i]=max_angspeed

    curr_q=curr_config[0:3]
    curr_arm_joints=curr_config[3:8]
    curr_arm_jointspeeds=control_vec[4:]

    curr_wheel_angles=curr_config[8:]
    curr_wheelspeeds=control_vec[0:4]

    curr_wheelspeeds=np.array(curr_wheelspeeds)


    F=np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                [ 1, 1, 1, 1],
                [ -1,1,-1, 1]])


    Vb=r/4 * F @ curr_wheelspeeds.T

    if Vb[0]==0:
        qb=np.array([0,Vb[1],Vb[2]])
    else:
        qb=np.array([Vb[0],
                    (Vb[1]*np.sin(Vb[0])+Vb[2]*(np.cos(Vb[0])-1))/Vb[0],
                    (Vb[2]*np.sin(Vb[0])+Vb[1]*(1-np.cos(Vb[0])))/Vb[0]])


    q_transform=np.array([[1, 0, 0],
                       [ 0, np.cos(curr_config[0]), -np.sin(curr_config[0])],
                       [ 0, np.sin(curr_config[0]), np.cos(curr_config[0])],
                       ])



    dq=q_transform @ qb

    newq=curr_q+dq*dt

    
  

    new_arm_joints=np.zeros((len(curr_arm_joints)))
    new_wheel_angles=np.zeros((len(curr_wheel_angles)))
    new_config=np.zeros(((len(curr_config))))



    for i in range(len(curr_arm_joints)):
        new_arm_joints[i]=curr_arm_joints[i]+curr_arm_jointspeeds[i]*dt

    for i in range(len(curr_wheel_angles)):
        new_wheel_angles[i]=curr_wheel_angles[i]+curr_wheelspeeds[i]*dt
    

    new_config[0:3]=newq
    new_config[3:8]=new_arm_joints
    new_config[8:]=new_wheel_angles


    return new_config

dt=0.01

# Initial state for the 'best' case
curr_state=[0.5,0.2,0.,0.,0.,0.,-1.5,0.,0.,0.,0.,0.]


final_traj=[]

Xerr_tot=0
Xerr_list=[]
Xerr_r=[]
Xerr_p=[]
Xerr_y=[]
Xerr_x=[]
Xerr_y=[]
Xerr_z=[]


for i in range(len(desired_traj)-1):

    """
    The main loop that cycles through each configuration in the desired 
    trajectory and calculates the requred configuration that the robot
    must follow to stay on the desired path. It generates a .csv file
    of the configurations that the robot must follow, as well as plots
    of the body twist errors. 

    The .csv files are used in CoppeliaSim for simualtion of the mobile robot. 
    """

    # Gains for 'best' case
    Kp=np.identity(6)*2.0
    Ki=np.identity(6)*0.05


    # Gains for 'newTask' case
    # Kp=np.identity(6)*2.0
    # Ki=np.identity(6)*0.05

    # Gains for 'overshoot' case
    # Kp=np.identity(6)*1.2
    # Ki=np.identity(6)*0.9 

    thetalist=curr_state[3:8]

    Tsb =  np.array([[np.cos(curr_state[0]), -np.sin(curr_state[0]),0, curr_state[1]],
                       [ np.sin(curr_state[0]), np.cos(curr_state[0]), 0, curr_state[2]],
                       [ 0, 0, 1, 0.0963],
                       [ 0, 0, 0, 1]])

    T0e=mr.FKinBody(M0e,Blist,thetalist)

    Tse=Tsb@ Tb0 @ M0e

    # Using the 1x12 current configuration state of the robot, calculate the current 4x4 SE(3) state X
    X=mr.FKinBody(Tse,Blist,thetalist)

    Xd=make_SE(desired_traj[i])

    Xd_next=make_SE(desired_traj[i+1])

    # Calculating the twist and errors for getting to the next desired state from the current state X
    V, Xerr, Xerr_tot=FeedbackControl(X,Xd,Xd_next,Xerr_tot,Kp,Ki,dt)

    Xerr_list.append(Xerr)
    Xerr_r.append(Xerr[0])
    Xerr_p.append(Xerr[1])
    Xerr_y.append(Xerr[2])
    Xerr_x.append(Xerr[3])
    Xerr_y.append(Xerr[4])
    Xerr_z.append(Xerr[5])


    J_base = mr.Adjoint(mr.TransInv(T0e)@mr.TransInv(Tb0))@(r/4*F)

    J_arm=mr.JacobianBody(Blist,thetalist)

    J=np.hstack((J_base,J_arm))

    # Calculating the speed control vector of wheel and arm joint speeds
    speed_control=np.linalg.pinv(J)@V

    # Calculating the configuration that the robot must actually follow to stay on desired path
    Next_state=NextState(curr_state,speed_control,dt,15.0)

    curr_state=Next_state

    # Add the calculated configuration to the final trajectory list

    if desired_traj[i][12]==1:

        # Append 1 or 0 determnes if the gripper is to be open or closed as determined by the 13th element of the desired traj vector

        final_traj.append(np.append(Next_state,1))
    else:
        final_traj.append(np.append(Next_state,0))

# Create .csv file
f1 = open("FinalTraj.csv", "w") 
np.savetxt("FinalTraj.csv", final_traj, delimiter = ",") 


# Plot errors

fig, ax=plt.subplots(2,3)

ax[0,0].plot(Xerr_r)
ax[0,1].plot(Xerr_p)
ax[0,2].plot(Xerr_y)
ax[1,0].plot(Xerr_x)
ax[1,1].plot(Xerr_y)
ax[1,2].plot(Xerr_z)

ax[0,0].set_title('Roll Error')
ax[0,1].set_title('Pitch Error')
ax[0,2].set_title('Yaw Error')


ax[1,0].set_title('X Error')
ax[1,1].set_title('Y Error')
ax[1,2].set_title('Z Error')

ax[0,0].set(xlabel='Time (s^-2)',ylabel='Error (rad)')
ax[0,1].set(xlabel='Time (s^-2)',ylabel='Error (rad)')
ax[0,2].set(xlabel='Time (s^-2)',ylabel='Error (rad)')


ax[1,0].set(xlabel='Time (s^-2)',ylabel='Error (m)')
ax[1,1].set(xlabel='Time (s^-2)',ylabel='Error (m)')
ax[1,2].set(xlabel='Time (s^-2)',ylabel='Error (m)')

fig, ax=plt.subplots()

plt.plot(Xerr_list)
ax.set_title('Error plot for Kp = 2.0 and Ki = 0.05')
ax.legend(['Roll', 'Pitch' , ' Yaw', 'X' , ' Y'  ,' Z'])
ax.set(xlabel='Time (s^-2)',ylabel='Error (m)')


plt.show()




    


















