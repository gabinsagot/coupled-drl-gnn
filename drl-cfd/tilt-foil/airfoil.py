# Generic imports
import os
import time
import numpy as np
import subprocess
# Import file for geometry creation
from geometry.mesh.Foil import * # type: ignore
# Import file for reward computation and mesh deformation
from avg_liftdrag import *
from idw import *

###############################################
### Panels in wind environment
class airfoil():

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'airfoil'
        self.base_folder = os.getcwd()
        self.path     = path
        self.act_size = 1 # Only the angle to change
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.shape_h  = 1.0
        self.x_min    = np.array([-1.0,]) # angles min 
        self.x_max    = np.array([ 1.0,]) # angles max
        self.x_0      = 2*np.random.rand((1))-1.0                      # initial position
        self.alpha_max = 30.0 # Remap to physical scale
        self.bad_rwrd = -2000
        self.cores    = '8'  #num of cores per env 
        self.dim      = '2d'
        self.dt       = 0.05 # using 1 because 3D uses varying dt, dt at the end is 0.1
        self.window   = [149.99,250.01] # Time window for drag/lift calculus. Corresponds to [175,200], but increment numbers differ due to varying dt in cfd
        self.timeout  = '3600'      # timeout limit in seconds (s) -> 1h

        # Set episode number
        self.episode  = 0
    
    ### CFD resolution
    def cfd_solve(self, x, ep):

        ## Create folders and copy cfd folder
        self.output_path       = self.path+'/'+str(ep)+'/'
        self.vtu_path          = self.output_path+'/vtu/'
        self.efforts_path   = self.output_path+'Efforts/'

        print("Base folder : ", self.base_folder)
        print("Self.path : ", self.path)

        os.makedirs(self.vtu_path, exist_ok= True)
        os.makedirs(self.efforts_path, exist_ok= True)
        os.system('cp -r cfd ' + self.base_folder + '/' + self.output_path + '.')

        self.write_actions(x,ep) # Saves action(s) to a file, already remapped to physical scale at this stage

        name = "object"
        try:
            self.surface = self.create_geometry(x, name, ep)
        except Exception as e:
            raise ValueError(f"ERROR: Geometry creation failed at episode {ep}: {e}. Probable self-intersecting surface, assigning bad reward")

        cfd_path = self.base_folder + '/' + self.output_path + 'cfd'

        with open(cfd_path + "/logCFD.out", "w") as log:
                subprocess.run(
                    [
                        "srun",
                        "--exclusive",
                        "-n", str(self.cores),
                        "-t", str(self.timeout),
                        str(Path("/home/gsagot/cimlib_CFD_driver")),
                        "lanceur/Principale.mtc"
                    ],
                    cwd=cfd_path,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True
                )

        ## Solve problem using cimlib and move vtu and drag folder
        cmd = (
            'cd ' + self.base_folder + '/' + self.output_path +
            'cfd && touch run.lock && mpirun -n ' + self.cores + ' --timeout ' + self.timeout +' '
            '' + self.base_folder + '/cimlib_CFD_driver lanceur/Principale.mtc > log.txt 2>&1'
        )
        #os.system(cmd)
        time.sleep(2)
        os.system('cp '+self.base_folder+'/'+self.output_path+'cfd/Resultats/*.txt '+self.base_folder+'/'+self.efforts_path+'.') # Copy the efforts.txt
        os.system('mv '+self.base_folder+'/'+self.output_path+'cfd/Resultats/'+self.dim+'/* '+self.base_folder+'/'+self.vtu_path+'.') # Move vtu.s
        # os.system('rm -r '+self.base_folder+'/'+self.output_path+'cfd') # Remove the copied cfd folder

        # Reward
        self.reward = self.compute_reward()
        self.write_rewards([self.reward],ep)
        print(f"Reward for episode {ep} : ", self.reward)

        ## Increment episode
        self.episode += 1

        return self.reward

    ### Take one step
    def step(self, actions, ep):

        self.write_actions_alt(actions,ep)

        conv_actions = self.convert_actions(actions)
        try:
            reward = self.cfd_solve(conv_actions, ep)
        except Exception as e:
            print("\n !!! cfd_solve() function failed !!!", e, flush=True)
            conv_actions = locals().get("conv_actions", np.zeros(self.act_size))
            return self.bad_rwrd, conv_actions
    
        return reward, conv_actions


    ### Convert actions
    def convert_actions(self, actions):

        # Convert actions
        conv_actions  = self.act_size*[None]

        for i in range(self.act_size):
            conv_actions[i] = actions[i]*self.alpha_max

        return conv_actions

    ### Provide observation
    def observe(self):
        # Always return the same observation
        return self.obs

    ### Close environment
    def close(self):
        pass
        

    def create_geometry(self, actions : np.ndarray, name : str, ep : int, plot : bool = False, naca0010 : bool = False):
        """
        Generates the mesh for the object at the given angle of attack and 
        copies the mesh in the right cfd directory
        
        Args: 
            actions: ArrayLike containing the actions to undertake
            name (str): name of the foil

        Returns: 
            float = foil's approximate area (polygon between points)

        Foil geometry generation:
        1) Writes all intermediates to geometry/mesh/{ep}/{txt,geo,msh,t}
        2) Produces t: geometry/mesh/{ep}/t/{name}_{ep}.t
        3) Copies  to results/.../0/{ep}/cfd/meshes/object.t
        """
        # Episode-local sandbox
        episode_root = os.path.join(self.base_folder, "geometry", "mesh", str(ep))
        txt_dir = os.path.join(episode_root, "txt")
        geo_dir = os.path.join(episode_root, "geo")
        msh_dir = os.path.join(episode_root, "msh")
        t_dir   = os.path.join(episode_root, "t")
        for d in (txt_dir, geo_dir, msh_dir, t_dir):
            os.makedirs(d, exist_ok=True)
            
        x_trans_domain = 2.5
        y_trans_domain = 2.0

        # Build the foil
        foil = Foil(10, 1.0, 1.0, work_dir=episode_root, suffix=f"_{ep}")
        foil.name = name  # 'object'
        foil.generate_airfoil_points(random=False)
        foil.apply_rotation(actions[0]/180.0*np.pi)
        foil.apply_translation(x_trans_domain, y_trans_domain) # Translate it where the boundary layer mesh is originally

        self.foil_area = foil.compute_surface() # Computes approximate area of the foil (polygons)
        if plot : foil.plot()
        
        # Save original foil points to compute displacements
        naca0010_foil = Foil(10, 1.0, 1.0, work_dir=episode_root, suffix=f"_{ep}")
        naca0010_foil.generate_airfoil_points(random = False)
        naca0010_foil.apply_translation(x_trans_domain, y_trans_domain) # Translate it where the boundary layer mesh is originally

        # Deform the original domain with IDW according to actions and control points position
        control_points = compute_idw_mesh(naca0010_foil, foil, ep, self.base_folder, self.path, interp_type="spline", p = 3)
        # Get every new control points & give it to foil.points()
        foil.points = control_points

        # Generate new .t file via sync()
        t_file_path = foil.sync()  # /geometry/mesh/{ep}/t/object_{ep}.t

        if not os.path.isfile(t_file_path):
            raise FileNotFoundError(f"Method foil.sync() did not create t-file at {t_file_path}")

        # Copy to results/.../0/{ep}/cfd/meshes/object.t
        meshes_dir = os.path.join(self.base_folder, self.path, str(ep), "cfd", "meshes")
        os.makedirs(meshes_dir, exist_ok=True)
        final_dst = os.path.join(meshes_dir, "object.t")
        tmp_dst = final_dst + ".tmp"

        # Copy to a tmp name, then rename to avoid partially written files
        shutil.copyfile(t_file_path, tmp_dst)
        os.replace(tmp_dst, final_dst)

        # Run mtcexe
        cmd = (
            f'cd "{meshes_dir}" && '
            f'module load cimlibxx/master && '
            f'echo 0 | mtcexe object.t > /dev/null 2>&1'
        )
        # os.system(f"bash -lc '{cmd}'")
        print("t_file copied and processed with mtc.")

        return foil.surface


    ### Reward Functions
    def compute_reward(self) -> float:
        """Compute the reward for the episode (ep) based on the forces data."""

        file_path = os.path.join(self.base_folder, f"{self.efforts_path}Efforts.txt")
        print(file_path)
        data = read_lift_drag(file_path)
        print(data)
        cx0_value, cy0_value = avg_lift_drag(data, plot=False)
        reward = cy0_value/cx0_value  # Maximize L/D

        return reward
    
        ### To write rotations at each environment
    def write_actions(self,actions,ep):
        with open(self.path+'/../actions.txt','a') as file:
            data_str = ' '.join( [str(ep)] + [str(rotation) for rotation in actions] )
            file.write(data_str + '\n')
        return('done writing actions of env '+str(ep)+' in file')

    ### To write rotations at each environment
    def write_actions_alt(self,actions,ep):
        with open('actions_alt.txt','a') as file:
            data_str = ' '.join( [str(ep)] + [str(rotation) for rotation in actions] )
            file.write(data_str + '\n')
        return('done writing actions of env '+str(ep)+' in file')

    ### To write rewards at each environment
    def write_rewards(self,rewards,ep):
        with open(self.path+'/../all_rewards.txt','a') as file:
            data_str = ' '.join( [str(ep)] + [str(val) for val in rewards] )
            file.write(data_str + '\n')
        return('done writing rewards of env '+str(ep)+' in file')

# ### Tests ###
# filepath = os.path.dirname(os.path.abspath(__file__))
# pbo = airfoil(filepath)
# action = np.array([-20.0])
# pbo.create_geometry(action,"object")
