import numpy as np
from math import nan, inf, pi, ceil, sqrt
from libdenavit import save_MASTAN2
from matplotlib import pyplot as plt


class Joist:
    def __init__(self, **attrs):
        
        # General information
        self.joist_name             = attrs.get('joist_name', '')  
        self.units                  = attrs.get('units', 'US')    # US: kips, in, ksi    
        
        # Overall geometry
        self.span                   = attrs['span']
        self.depth                  = attrs['depth']
        self.truss_type             = attrs['truss_type']  # e.g., 'Warren' or 'ModifiedWarren'
        self.top_chord_panel_lengths = attrs['top_chord_panel_lengths']
        self.bottom_chord_panel_lengths = attrs['bottom_chord_panel_lengths']
        if self.units == 'US':
            self.support_location   = attrs.get('support_location',2)
        else:
            self.support_location   = attrs['support_location']
        self.top_chord_extension_length_left = attrs.get('top_chord_extension_length_left', 0)
        self.top_chord_extension_length_right = attrs.get('top_chord_extension_length_right', 0)
        
        # Member information
        self.sections               = attrs['sections']  # Dictionary of sections
        self.top_chord_section      = attrs['top_chord_section']  # String 
        self.bottom_chord_section   = attrs['bottom_chord_section']  # String
        self.web_sections           = attrs['web_sections'] # List of strings

        # Material information
        self.material_name              = attrs.get('material_name', 'Steel')       # @new  
        if self.units == 'US':                                
            self.E                  = attrs.get('E', 29000)
            self.Fy                 = attrs.get('Fy', 50)
            self._top_chord_Fy      = attrs.get('top_chord_Fy', None)
            self._bottom_chord_Fy   = attrs.get('bottom_chord_Fy', None)
            self._web_Fy            = attrs.get('web_Fy', None)
            self.v                  = attrs.get('v', 0.3)
            self.unit_weight        = attrs.get('unit_weight', 490/1728/1000)
        else:
            self.E                  = attrs['E']
            self.Fy                 = attrs['Fy']
            self._top_chord_Fy      = attrs['top_chord_Fy']
            self._bottom_chord_Fy   = attrs['bottom_chord_Fy']
            self._web_Fy            = attrs['web_Fy']            
            self.v                  = attrs['v']
            self.unit_weight        = attrs['unit_weight']

        # Loads
        self.uniform_load           = attrs.get('uniform_load', 0)
        self.lateral_pressure       = attrs.get('lateral_pressure', 0)
        self.axial_load               = attrs.get('axial_load', 0)     # Compression (+), applied at right support
        
        # Analysis Options
        self.neglect_self_weight    = attrs.get('neglect_self_weight', True)
        self.pin_web_members        = attrs.get('pin_web_members', False)
        self.model_chords_as_double_angles = attrs.get('model_chords_as_double_angles', False) # @todo
        self.bottom_chord_extension_length = attrs.get('bottom_chord_extension_length', 0)
        self.maximum_element_length = attrs.get('maximum_element_length', None)
        self._maximum_top_chord_element_length = attrs.get('maximum_top_chord_element_length', None)
        self._maximum_bottom_chord_element_length = attrs.get('maximum_bottom_chord_element_length', None)
        self._maximum_web_member_element_length = attrs.get('maximum_web_member_element_length', None)

    # Section Retreival Functions
    @property
    def top_chord(self):
        return self.sections[self.top_chord_section]
    
    @property
    def bottom_chord(self):
        return self.sections[self.bottom_chord_section]

    # Element Length Functions
    @property
    def maximum_top_chord_element_length(self):
        if self._maximum_top_chord_element_length is not None:
            return self._maximum_top_chord_element_length
        else:
            return self.maximum_element_length

    @maximum_top_chord_element_length.setter
    def maximum_top_chord_element_length(self, value):
        self._maximum_top_chord_element_length = value

    @property
    def maximum_bottom_chord_element_length(self):
        if self._maximum_bottom_chord_element_length is not None:
            return self._maximum_bottom_chord_element_length
        else:
            return self.maximum_element_length

    @maximum_bottom_chord_element_length.setter
    def maximum_bottom_chord_element_length(self, value):
        self._maximum_bottom_chord_element_length = value

    @property
    def maximum_web_member_element_length(self):
        if self._maximum_web_member_element_length is not None:
            return self._maximum_web_member_element_length
        else:
            return self.maximum_element_length

    @maximum_web_member_element_length.setter
    def maximum_web_member_element_length(self, value):
        self._maximum_web_member_element_length = value

    # Yield Strength Functions
    @property
    def top_chord_Fy(self):
        if self._top_chord_Fy is not None:
            return self._top_chord_Fy
        else:
            return self.Fy
    
    @top_chord_Fy.setter
    def top_chord_Fy(self, value):
        self._top_chord_Fy = value

    @property
    def bottom_chord_Fy(self):
        if self._bottom_chord_Fy is not None:
            return self._bottom_chord_Fy
        else:
            return self.Fy
    
    @bottom_chord_Fy.setter
    def bottom_chord_Fy(self, value):
        self._bottom_chord_Fy = value

    @property
    def web_Fy(self):
        if self._web_Fy is not None:
            return self._web_Fy
        else:
            return self.Fy
    
    @web_Fy.setter
    def web_Fy(self, value):
        self._web_Fy = value

    # Physical Joist Properties
    def effective_depth(self):
        return self.depth - self.top_chord.y_bar - self.bottom_chord.y_bar

    # Centerline, members-only weight, excluding seats, clips, and weld metal
    def total_weight(self):
        self.build_joist_geometry()
        W = 0

        #Top Chord
        for i in range(self._number_of_top_chord_joints - 1):
            L = self._joint_coords[i+1][0] - self._joint_coords[i][0]
            W += L * self.top_chord.A * self.unit_weight

        #Bottom Chord
        start = self._number_of_top_chord_joints
        for i in range(start, start + self._number_of_bottom_chord_joints - 1):
            L = self._joint_coords[i+1][0] - self._joint_coords[i][0]
            W += L * self.bottom_chord.A * self.unit_weight

        #Web Members
        web_section_list = self.web_sections_all()
        for i, web_joints in enumerate(self._web_connectivity):
            ix = self._joint_coords[web_joints[0]-1][0]
            iy = self._joint_coords[web_joints[0]-1][1]
            jx = self._joint_coords[web_joints[1]-1][0]
            jy = self._joint_coords[web_joints[1]-1][1]
            L = sqrt((jx-ix)**2 + (jy-iy)**2)
            W += L * self.sections[web_section_list[i]].A * self.unit_weight

        return W

    def weight_per_foot(self):
        return self.total_weight() / (self.span/12)
    
    # Attribut Parsing/Expanding Functions
    def top_chord_panel_point_x_coords(self,include_ends=True,include_supports=True):
        x_coords = [0]
        if sum(self.top_chord_panel_lengths) == self.span:
            for length in self.top_chord_panel_lengths:
                x_coords.append(x_coords[-1]+length)
        elif sum(self.top_chord_panel_lengths[0:-1]) <= self.span/2:
            remaining_length = self.span - 2*sum(self.top_chord_panel_lengths[0:-1])
            num_middle_panels = remaining_length/self.top_chord_panel_lengths[-1]

            if not num_middle_panels.is_integer():
                raise ValueError('Invalid self.top_chord_panel_lengths: num_middle_panels not an integer')
            
            for length in self.top_chord_panel_lengths:
                x_coords.append(x_coords[-1]+length)
            for i in range(int(num_middle_panels)-2):
                x_coords.append(x_coords[-1]+self.top_chord_panel_lengths[-1])
            for length in reversed(self.top_chord_panel_lengths):
                x_coords.append(x_coords[-1] + length)
        else:
            raise ValueError('Invalid self.top_chord_panel_lengths')
        
        # Adjust for top chord extensions
        x_coords[0] -= self.top_chord_extension_length_left
        x_coords[-1] += self.top_chord_extension_length_right
        
        # Add support nodes if wanted
        if include_supports:
            # Add node at supports at each end, if necessary and requested. 
            if self.support_location < self.top_chord_panel_lengths[0]:
                x_coords.insert(1, self.support_location)
                x_coords.insert(-1, self.span-self.support_location)
            
            elif self.support_location == self.top_chord_panel_lengths[0]:
                pass
            
            elif self.support_location < self.top_chord_panel_lengths[0]+self.top_chord_panel_lengths[1]:
                x_coords.insert(2, self.support_location)
                x_coords.insert(-2, self.span-self.support_location)
            
            else: 
                raise ValueError('Invalid combination of support_location and top_chord_panel_lengths')

        # Remove end nodes if not wanted
        if not include_ends:
            x_coords = x_coords[1:-1]

        # Make sure all the entrys are floats
        x_coords = [float(i) for i in x_coords]

        return x_coords               

    def bottom_chord_panel_point_x_coords(self):
        x_coords = [0]
        
        if sum(self.bottom_chord_panel_lengths) == self.span:
            for length in self.bottom_chord_panel_lengths:
                x_coords.append(x_coords[-1]+length)
        elif sum(self.bottom_chord_panel_lengths[0:-1]) <= self.span/2:         
            remaining_length = self.span - 2*sum(self.bottom_chord_panel_lengths[0:-1])          
            num_middle_panels = remaining_length/self.bottom_chord_panel_lengths[-1]

            if not num_middle_panels.is_integer():
                raise ValueError('Invalid self.bottom_chord_panel_lengths: num_middle_panels not an integer')
            
            for length in self.bottom_chord_panel_lengths[0:-1]:
                x_coords.append(x_coords[-1]+length)
            for i in range(int(num_middle_panels)):
                x_coords.append(x_coords[-1]+self.bottom_chord_panel_lengths[-1])
            for length in reversed(self.bottom_chord_panel_lengths[0:-1]):
                x_coords.append(x_coords[-1] + length)

        else:
            raise ValueError('Invalid self.bottom_chord_panel_lengths')

        # Trim x_coords to remove values at 0 and span
        x_coords = x_coords[1:-1]

        # Add bottom chord extension (if defined)
        if self.bottom_chord_extension_length > 0:
            x_coords.insert(0, x_coords[0] - self.bottom_chord_extension_length)
            x_coords.append(x_coords[-1] + self.bottom_chord_extension_length)

        x_coords = [float(i) for i in x_coords]

        return x_coords

    def numWebMembers(self):
        if self.bottom_chord_extension_length > 0:
            if self.truss_type.lower() == 'warren':
                return (len(self.bottom_chord_panel_point_x_coords()) - 1) * 2
            elif self.truss_type.lower() == 'modifiedwarren':
                return (len(self.bottom_chord_panel_point_x_coords()) - 2) * 3
            else:
                raise ValueError('Truss type not implemented at this time')
        else:
            if self.truss_type.lower() == 'warren':
                return (len(self.bottom_chord_panel_point_x_coords()) + 1) * 2
            elif self.truss_type.lower() == 'modifiedwarren':
                return (len(self.bottom_chord_panel_point_x_coords())) * 3
            else:
                raise ValueError('Truss type not implemented at this time')

    def web_sections_all(self):        
        number_of_web_members = self.numWebMembers()
        
        if len(self.web_sections) == number_of_web_members:
            return self.web_sections.copy()
        elif 2*len(self.web_sections)-1 == number_of_web_members:
            web_sections = self.web_sections.copy()
            web_sections.extend(reversed(self.web_sections.copy()[:-1]))
            return web_sections
        elif 2*len(self.web_sections) <= number_of_web_members:
            web_sections = self.web_sections.copy()
            repeats = number_of_web_members - 2*len(web_sections)
            web_sections.extend([web_sections[-1]]*repeats)
            web_sections.extend(reversed(self.web_sections.copy()))
            return web_sections
        else:
            raise ValueError(f'Web configuration must have a length less than or equal to half the number of web members')
   
    def is_support_at_panel_point(self):
        return self.support_location == self.top_chord_panel_lengths[0]

    # Strength Calculation Helpers
    def top_chord_shear_check_locations(self):
        tc_panel_point_x_coords = self.top_chord_panel_point_x_coords()
        if self.truss_type.lower() == 'warren':
            if self.is_support_at_panel_point():
                Lss = tc_panel_point_x_coords[3:-3]
            else:
                Lss = tc_panel_point_x_coords[4:-4]
        elif self.truss_type.lower() == 'modifiedwarren':
            if self.is_support_at_panel_point():
                Lss = tc_panel_point_x_coords[3:-3:2]
            else:
                Lss = tc_panel_point_x_coords[4:-4:2]
        else:
            raise ValueError(f'Unknown truss_type: {self.truss_type}')
        return Lss

    def bottom_chord_shear_check_locations(self):
        bc_panel_point_x_coords = self.bottom_chord_panel_point_x_coords()
        if self.truss_type.lower() == 'warren':
            if self.bottom_chord_extension_length > 0:
                Lss = bc_panel_point_x_coords[1:-1]
            else:
                Lss = bc_panel_point_x_coords
        elif self.truss_type.lower() == 'modifiedwarren':
            if self.bottom_chord_extension_length > 0:
                Lss = bc_panel_point_x_coords[1:-1]
            else:
                Lss = bc_panel_point_x_coords
        else:
            raise ValueError(f'Unknown truss_type: {self.truss_type}')
        return Lss

    def build_joist_geometry(self):
        
        de = self.effective_depth()       
        tc_x_coords = self.top_chord_panel_point_x_coords()
        bc_x_coords = self.bottom_chord_panel_point_x_coords()        

        # Define Joints (i.e., panel points, chord ends, and supports)
        joint_coords = []
        for x_coord in tc_x_coords:
            joint_coords.append([x_coord, 0, 0])
        for x_coord in bc_x_coords:
            joint_coords.append([x_coord, -de, 0])

        joist_indicies = list(range(1, len(joint_coords) + 1))
        
        number_of_top_chord_joints = len(tc_x_coords)
        number_of_bottom_chord_joints = len(bc_x_coords)
        
        if self.support_location < self.top_chord_panel_lengths[0]:
            left_support_joint_index = 2
            right_support_joint_index = number_of_top_chord_joints-1
            top_chord_panel_point_indicies = list(range(3,number_of_top_chord_joints-1))
        elif self.support_location == self.top_chord_panel_lengths[0]:
            left_support_joint_index = 2
            right_support_joint_index = number_of_top_chord_joints-1
            top_chord_panel_point_indicies = list(range(2,number_of_top_chord_joints))
        elif self.support_location < self.top_chord_panel_lengths[0]+self.top_chord_panel_lengths[1]:
            left_support_joint_index = 3
            right_support_joint_index = number_of_top_chord_joints-2            
            top_chord_panel_point_indicies = [i for i in range(2, number_of_top_chord_joints) if i not in (3, number_of_top_chord_joints - 2)]
        else: 
            raise ValueError('Invalid combination of support_location and top_chord_panel_lengths')
            
        # Web Member Connectivity
        web_connectivity = []
        if self.bottom_chord_extension_length > 0:
            bc_start = number_of_top_chord_joints + 2
            bc_end = number_of_top_chord_joints + number_of_bottom_chord_joints - 1
        else:
            bc_start = number_of_top_chord_joints + 1
            bc_end = number_of_top_chord_joints + number_of_bottom_chord_joints
        
        if self.truss_type.lower() == 'warren':
            tc_i = 0
            for bc_i in range(bc_start,bc_end+1):
                if (bc_i == bc_start) or (bc_i == bc_end):
                    web_connectivity.append([top_chord_panel_point_indicies[tc_i], bc_i])
                    web_connectivity.append([top_chord_panel_point_indicies[tc_i+1], bc_i])
                    web_connectivity.append([top_chord_panel_point_indicies[tc_i+2], bc_i])
                    tc_i += 2
                else:
                    web_connectivity.append([top_chord_panel_point_indicies[tc_i], bc_i])
                    web_connectivity.append([top_chord_panel_point_indicies[tc_i+1], bc_i])
                    tc_i += 1
                    
        elif self.truss_type.lower() == 'modifiedwarren':
            tc_i = 0
            for bc_i in range(bc_start,bc_end+1):
                web_connectivity.append([top_chord_panel_point_indicies[tc_i], bc_i])
                web_connectivity.append([top_chord_panel_point_indicies[tc_i+1], bc_i])
                web_connectivity.append([top_chord_panel_point_indicies[tc_i+2], bc_i])
                tc_i += 2

        else:
            raise ValueError(f'Unknown truss_type: {self.truss_type}')
            
        # Store Results
        self._joint_coords = joint_coords
        self._web_connectivity = web_connectivity
        self._number_of_top_chord_joints = number_of_top_chord_joints
        self._number_of_bottom_chord_joints = number_of_bottom_chord_joints
        self._left_support_joint_index = left_support_joint_index
        self._right_support_joint_index = right_support_joint_index


    def saveMASTAN2(self, model_title=None):
        self.build_joist_geometry()
        de = self.effective_depth()
        node_info = self._joint_coords
               
        # Section Information
        sect_info=[]
        sect_name=[]
        for name, section in self.sections.items():
            sect_info.append(section.MASTAN2_sect_info())
            sect_name.append(name)

        # Initilize Element and Uniform Load Arrays
        elem_info = []
        uniload_info = []
        
        # Define Top Chord Elements
        tc_x_coords = self.top_chord_panel_point_x_coords()
        section_index = sect_name.index(self.top_chord_section) + 1
        lateral_load = self.lateral_pressure*self.sections[self.top_chord_section].in_plane_depth()

        for i in range(self._number_of_top_chord_joints-1):
            
            # Determine number of elements in panel
            if self.maximum_top_chord_element_length == None or self.maximum_top_chord_element_length == 0:
                num_elements = 1
            else:
                panel_length = tc_x_coords[i+1] - tc_x_coords[i]
                num_elements = ceil(panel_length/self.maximum_top_chord_element_length)
            
            # Define elements (and added nodes if necessary)
            i_node = i+1
            for j in range(num_elements):
                
                if j == num_elements-1:
                    j_node = i+2
                else:
                    node_info.append([tc_x_coords[i] + (j+1)/num_elements*panel_length, 0, 0])
                    j_node = len(node_info)
                
                elem_info.append([i_node, j_node, section_index, 1, 0, 0, 0, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])
                uniload_info.append([0, -self.uniform_load, lateral_load])    
                
                i_node = j_node                
        
        # Define Bottom Chord Elements
        bc_x_coords = self.bottom_chord_panel_point_x_coords()
        section_index = sect_name.index(self.bottom_chord_section) + 1    
        lateral_load = self.lateral_pressure*self.sections[self.bottom_chord_section].in_plane_depth()

        for i in range(self._number_of_bottom_chord_joints-1):
            
            # Determine number of elements in panel
            if self.maximum_bottom_chord_element_length == None or self.maximum_bottom_chord_element_length == 0:
                num_elements = 1
            else:
                panel_length = bc_x_coords[i+1] - bc_x_coords[i]
                num_elements = ceil(panel_length/self.maximum_bottom_chord_element_length)
            
            # Define elements (and added nodes if necessary)
            i_node = self._number_of_top_chord_joints+i+1
            for j in range(num_elements):
                
                if j == num_elements-1:
                    j_node = self._number_of_top_chord_joints+i+2
                else:
                    node_info.append([bc_x_coords[i] + (j+1)/num_elements*panel_length, -de, 0])
                    j_node = len(node_info)
                
                elem_info.append([i_node, j_node, section_index, 2, 0, 0, 0, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])
                uniload_info.append([0, 0, lateral_load])    
                
                i_node = j_node  

        # Define Web Elements
        if self.pin_web_members:
            pinned = 1
        else:
            pinned = 0
        web_section_list = self.web_sections_all()

        for i, web_joints in enumerate(self._web_connectivity):
            ix = self._joint_coords[web_joints[0]-1][0]
            iy = self._joint_coords[web_joints[0]-1][1]
            jx = self._joint_coords[web_joints[1]-1][0]
            jy = self._joint_coords[web_joints[1]-1][1]
            
            if ix > jx:
                beta = pi
            else:
                beta = 0
            section_index = sect_name.index(web_section_list[i]) + 1    
            lateral_load = self.lateral_pressure*self.sections[web_section_list[i]].in_plane_depth()

            # Determine number of elements in panel
            if self.maximum_web_member_element_length == None or self.maximum_web_member_element_length == 0:
                num_elements = 1
            else:
                web_length = sqrt((jx-ix)**2 + (jy-iy)**2)
                num_elements = ceil(web_length/self.maximum_web_member_element_length)
                
            # Define elements (and added nodes if necessary)
            if num_elements == 1:
                elem_info.append([web_joints[0], web_joints[1], section_index, 3, beta, pinned, pinned, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])
                uniload_info.append([0, 0, lateral_load])    

            else:
            
                i_node = web_joints[0]
                for j in range(num_elements):
                    
                    if j == num_elements-1:
                        j_node = web_joints[1]
                    else:
                        node_info.append([ix + (j+1)/num_elements*(jx-ix), iy + (j+1)/num_elements*(jy-iy), 0])
                        j_node = len(node_info)
                    
                    if j == 0:
                        elem_info.append([i_node, j_node, section_index, 3, beta, 0, pinned, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])
                    elif j == num_elements-1:
                        elem_info.append([i_node, j_node, section_index, 3, beta, pinned, 0, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])
                    else:
                        elem_info.append([i_node, j_node, section_index, 3, beta, 0, 0, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])

                    uniload_info.append([0, 0, lateral_load])    
                    
                    i_node = j_node 

        # Support Information
        num_nodes = len(node_info)
        support_info = np.zeros([num_nodes,6])
        support_info[:] = nan
        support_info[self._left_support_joint_index-1] = [0, 0, 0, 0, nan, nan]
        support_info[self._right_support_joint_index-1] = [nan, 0, 0, 0, nan, nan]

        # Nodal Loads
        nload_info = np.zeros([num_nodes,6])
        nload_info[self._right_support_joint_index-1][0] = -self.axial_load   
        
        # Material Information
        mat_info = [[self.E, self.v, self.top_chord_Fy, self.unit_weight],
                    [self.E, self.v, self.bottom_chord_Fy, self.unit_weight],
                    [self.E, self.v, self.web_Fy, self.unit_weight]]
        
        if self.neglect_self_weight: 
            # Set unit weight to zero
            for mat in mat_info:
                mat[3] = 0
        
        mat_name = ['top_chord_' + self.material_name, 
                    'bottom_chord_' + self.material_name, 
                    'web_' + self.material_name]

        # Save Joist     
        if model_title is None:
            model_title = self.joist_name
               
        save_MASTAN2(model_title=model_title, 
                     node_info=node_info, 
                     elem_info=elem_info,
                     support_info=support_info,
                     nload_info=nload_info, 
                     uniload_info=uniload_info, 
                     sect_info=sect_info,
                     sect_name=sect_name, 
                     mat_info=mat_info,
                     mat_name=mat_name)

    # Plotting Functions
    def plot_joist(self):
        self.build_joist_geometry()
        tc_x = self.top_chord_panel_point_x_coords()
        bc_x = self.bottom_chord_panel_point_x_coords()
        de = self.effective_depth()

        # Plot chords
        plt.plot(tc_x, [0]*len(tc_x), 'k')
        plt.plot(bc_x, [-de]*len(bc_x), 'k')
        for i in self._joint_coords:
            plt.plot(i[0], i[1], 'ko')

        # Plot web members
        web_section_list = self.web_sections_all()
        for i, joints in enumerate(self._web_connectivity):
            ix = self._joint_coords[joints[0]-1][0]
            iy = self._joint_coords[joints[0]-1][1]
            jx = self._joint_coords[joints[1]-1][0]
            jy = self._joint_coords[joints[1]-1][1]
            plt.plot([ix, jx], [iy, jy], 'k')
            plt.text((ix+jx)/2, (iy+jy)/2, web_section_list[i], fontsize=8, ha='center', va='center')

        plt.ylim([-self.depth*1.1, 0.5])
        plt.show()
