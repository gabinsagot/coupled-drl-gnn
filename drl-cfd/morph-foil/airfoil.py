# Generic imports
import os
import shutil
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
        self.act_size = 9
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.shape_h  = 1.0
        self.x_min    = -np.ones(self.act_size)     # Generates action-sized array filled with 0 
        self.x_max    = np.ones(self.act_size)      # Generates action-sized array filled with 1.0 
        self.x_0      = np.array([np.random.rand((1)) for i in range(self.act_size)])   # initial action

        # Remap to physical scale:
        self.physical_scale = np.array([0.1, 0.15, 0.15, 0.15, 0.1,      # Camber limits
                                        0.2, 0.2, 0.2,                  # Thickness limits (everything except close to trailing and leading edge)
                                        45])                            # Rotation limit
        
        self.bad_rwrd = -10.0
        self.cores    = '8'  #num of cores per env 
        self.dim      = '2d'
        self.timeout  = '3600'      # timeout limit in seconds (s) -> 1h

        self.foil_area = 0.0                        # Airfoil area initialization

        # Set episode number
        self.episode  = 0
    
    ### CFD resolution
    def cfd_solve(self, x, ep):

        ## Create folders and copy cfd folder
        self.output_path    = self.path+'/'+str(ep)+'/'
        self.vtu_path       = self.output_path+'/vtu/'
        self.efforts_path   = self.output_path+'Efforts/'
        self.geometries_path       = self.output_path+'/geometries/'

        os.makedirs(self.vtu_path, exist_ok= True)
        os.makedirs(self.efforts_path, exist_ok= True)
        os.makedirs(self.geometries_path, exist_ok= True)
        os.system('cp -r cfd ' + self.base_folder + '/' + self.output_path + '.')

        self.write_actions(x,ep) # Saves action(s) to a file, already remapped to physical scale at this stage

        # Try to build geometry. If it fails, raise error and exit cfd_solve to assign bad reward
        try:
            name = "object"
            self.surface = self.create_geometry(x, name, ep)
        except Exception as e:
            raise ValueError(f"ERROR: Geometry creation failed at episode {ep}: {e}. Assigning bad reward")
        
        t_file_path = os.path.join(self.base_folder, self.output_path, 'cfd/meshes/object.t')
        if not os.path.isfile(t_file_path):
            print(f"WARNING : The final .t file is not found in {t_file_path}")
            raise ValueError(f"Failed to materialize {t_file_path}")

        ## Solve problem using cimlib and move vtu and drag folder
        cfd_path = self.base_folder + '/' + self.output_path + 'cfd'
        try:
            with open(cfd_path + "/logCFD.txt", "w") as log:
                subprocess.run(
                    [
                        "srun",
                        "--exclusive",
                        "-n", str(self.cores),
                        "-t", str(self.timeout),
                        str(self.base_folder + "/cimlib_CFD_driver"),
                        "lanceur/Principale.mtc"
                    ],
                    cwd=cfd_path,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=int(self.timeout)
                )
        except Exception as e:
            raise ValueError(f"ERROR: CFD simulation did not start at episode {ep}: {e}. Check {cfd_path}/logCFD.txt for details.")

        time.sleep(2)
        os.system('cp '+self.base_folder+'/'+self.output_path+'cfd/Resultats/*.txt '+self.base_folder+'/'+self.efforts_path+'.') # Copy the efforts.txt
        os.system('mv '+self.base_folder+'/'+self.output_path+'cfd/Resultats/'+self.dim+'/* '+self.base_folder+'/'+self.vtu_path+'.') # Move vtu.s
        os.system('mv '+self.base_folder+'/geometry/mesh/' + f'{ep}' + '/geo/* '+self.base_folder+'/'+self.geometries_path+'.') # Move object.geo
        os.system('mv '+self.base_folder+'/'+self.output_path+'cfd/meshes/* '+self.base_folder+'/'+self.geometries_path+'.') # Move object.t, .geo and domain.t

        os.system('rm -r '+self.base_folder+'/'+self.output_path+'cfd') # Remove the copied cfd folder
        os.system('rm -r '+self.base_folder+'/geometry/mesh/' + f'{ep}') # Remove the episode folder in geometry

        # Reward
        self.reward = self.compute_reward()
        self.write_rewards([self.reward],ep)
        # print(f"Reward for episode {ep} : ", self.reward)

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
            print("\n ERROR in cfd_solve(): ", e, flush=True)
            conv_actions = locals().get("conv_actions", np.zeros(self.act_size))
            self.reward = self.bad_rwrd
            self.write_rewards([self.reward],ep)

        reward = self.reward
        return reward, conv_actions


    ### Convert actions
    def convert_actions(self, actions):

        # Actions are taken in [-1;1], so transform according to expected action form in apply_xxx foil methods
        # Positive camber values (more or less concave): remap to [0.05, 1]
        actions[:5] = (0.45*actions[:5])+0.55        
        # Positive thicknesses: remap to [0.05, 1]
        actions[5:-1] = (0.45*actions[5:-1])+0.55  
        # Negative rotation to generate lift
        actions[-1] = 0.5*(actions[-1]-1.0)
        # Convert actions
        #print("Actions remapped avant physical scale : ", actions)
        conv_actions  = np.multiply(actions, self.physical_scale)
        # print("Actions converties au physical scale : ", conv_actions)

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
        foil = Foil(10, 1.0, 1.0, work_dir=episode_root, name = name, suffix=f"_{ep}")
        foil.name = name  # 'object'
        foil.generate_airfoil_points(random=False)
        foil.apply_camber_thickness(actions) # Apply deformation actions

        foil.apply_translation(x_trans_domain, y_trans_domain) # Translate it where the boundary layer mesh is originally

        self.foil_area = foil.compute_surface() # Computes approximate area of the foil (polygons)
        if plot : foil.plot()
        
        # Save original foil points to compute displacements
        naca0010_foil = Foil(10, 1.0, 1.0, work_dir=episode_root, suffix=f"_{ep}")
        naca0010_foil.generate_airfoil_points(random = False)
        naca0010_foil.apply_translation(x_trans_domain, y_trans_domain) # Translate it where the boundary layer mesh is originally

        # Deform the original domain with IDW according to actions and control points position
        control_points = compute_idw_mesh(naca0010_foil, foil, ep, self.base_folder, self.path, interp_type="bezier", density = 100, p=6, a=1e-10)
        # Get every new control points & give it to foil.points()
        foil.points = control_points

        # Generate new .t file via get_mesh and get_t
        try :
            geo_path = foil.get_geo()
            msh_path = foil.get_mesh_timeout(geo_path, timeout=5)
        except Exception as e:
            raise RuntimeError(f"Unable to mesh geometry at episode {ep} : {e}") from e
        try :
            t_file_path = os.path.join(episode_root, "t", f"{name}_{ep}.t")
            foil.convert_gmsh_to_mtc(msh_path, t_file_path, False)  # /geometry/mesh/{ep}/t/object_{ep}.t
        except Exception as e:
            raise RuntimeError(f"Unable to build .t file at episode {ep} : {e}")
        
        try :
            # Copy to results/.../0/{ep}/cfd/meshes/object.t
            meshes_dir = os.path.join(self.base_folder, self.path, str(ep), "cfd", "meshes")
            os.makedirs(meshes_dir, exist_ok=True)
            final_dst = os.path.join(meshes_dir, "object.t")
            tmp_dst = final_dst + ".tmp"

            # Copy to a tmp name, then rename to avoid partially written files
            shutil.copyfile(t_file_path, tmp_dst)
            os.replace(tmp_dst, final_dst)

        finally :
            if not os.path.isfile(final_dst):
                print(f"WARNING : The final name.t is not found in {final_dst}", flush=True)
                raise FileNotFoundError(f"Failed to materialize {final_dst}")

        # Run mtcexe
        cmd = (
            f'cd "{meshes_dir}" && '
            f'module load cimlibxx/master && '
            f'echo 0 | mtcexe object.t > /dev/null 2>&1'
        )
        # os.system(f"bash -lc '{cmd}'")
        # print("t_file copied and processed with mtc.")

        return foil.surface


    ### Reward Functions
    def compute_reward(self) -> float:
        """Compute the reward for the episode (ep) based on the forces data."""

        file_path = os.path.join(self.base_folder, f"{self.efforts_path}Efforts.txt")
        try:
            data = read_lift_drag(file_path)
            cx0_value, cy0_value = avg_lift_drag(data, plot=False)
        except Exception as e:
            raise ValueError(f"ERROR: Reward computation failed at episode {ep}: {e}.")
        sface_penalty = np.abs(0.100-self.surface) # Area gap to target area
        reward = 10*(np.sign(cy0_value)*np.power(np.abs(cy0_value), 3/2)/cx0_value - 2*sface_penalty)  # Maximise foil endurance under area constraints

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
    
