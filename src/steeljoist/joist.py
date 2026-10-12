import numpy as np
from math import nan, inf, pi, ceil, dist
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
            self.unit_weight        = attrs.get('unit_weight', 490/1728)
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
            
            elif (self.support_location > self.top_chord_panel_lengths[0]) and (self.support_location < self.top_chord_panel_lengths[0]+self.top_chord_panel_lengths[1]):
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

    # MASTAN2 Functions
    def TopNodeInfo(self):
        node_info = []
        tc_x_coords = self.top_chord_panel_point_x_coords()
        for x_coord in tc_x_coords:
            node_info.append([x_coord, 0, 0])
        return node_info
    
    def TopIndexInfo(self):
        return list(range(len(self.TopNodeInfo())+1))[1:]

    def BotNodeInfo(self):
        node_info = []
        de = self.effective_depth()       
        bc_x_coords = self.bottom_chord_panel_point_x_coords()
        for x_coord in bc_x_coords:
            node_info.append([x_coord, -de, 0])
        return node_info
    
    def BotIndexInfo(self):
        return list(range(len(self.bottom_chord_panel_point_x_coords())+len(self.TopIndexInfo())+1))[1+len(self.TopIndexInfo()):]

    def TopChordInfo(self):
        # node_info returns only the intermediate nodes defined by the maximum panel point length
        node_info = []
        ele_info = []
        uniload_info = []

        tc_x_coords = self.top_chord_panel_point_x_coords()
        curMax = max(self.BotIndexInfo())
        expanded_top_index_info = self.TopIndexInfo()
        secIndex = self.SectionInfo()[1].index(self.top_chord_section) + 1

        if self.maximum_top_chord_element_length != 0 and self.maximum_top_chord_element_length != None:
            for i, j in enumerate(tc_x_coords[:-1]):
                diff = tc_x_coords[i+1] - tc_x_coords[i]
                fac = ceil(diff/self.maximum_top_chord_element_length)
                if fac != 1:
                    for k in list(range(fac))[1:]:
                        node_info.append([j + (k*diff/fac), 0, 0])
                        expanded_top_index_info.insert(expanded_top_index_info.index(i+1)+k, curMax + k)
                    curMax = max(expanded_top_index_info)
        for i in range(len(expanded_top_index_info[:-1])):
            ele_info.append([expanded_top_index_info[i], expanded_top_index_info[i+1], secIndex, 1, 0, 0, 0, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])
            uniload_info.append([0, -self.uniform_load, self.LateralLoad(self.top_chord_section)])
        return node_info, expanded_top_index_info, ele_info, uniload_info
    
    def BotChordInfo(self):
        node_info = []
        ele_info = []
        uniload_info = []

        bc_x_coords = self.bottom_chord_panel_point_x_coords()
        curMax = max(self.TopChordInfo()[1])
        expanded_bot_index_info = self.BotIndexInfo()
        secIndex = self.SectionInfo()[1].index(self.bottom_chord_section) + 1    

        de = self.effective_depth()       
        if self.maximum_bottom_chord_element_length != 0 and self.maximum_bottom_chord_element_length != None:
            for i, j in enumerate(bc_x_coords[:-1]):
                diff = bc_x_coords[i+1] - bc_x_coords[i]
                fac = ceil(diff/self.maximum_bottom_chord_element_length)
                if fac != 1:
                    for k in list(range(fac))[1:]:
                        node_info.append([j + (k*diff/fac), -de, 0])
                        expanded_bot_index_info.insert(expanded_bot_index_info.index(i+1+max(self.TopIndexInfo()))+k, curMax + k)
                    curMax = max(expanded_bot_index_info)
        for i in range(len(expanded_bot_index_info[:-1])):
            ele_info.append([expanded_bot_index_info[i], expanded_bot_index_info[i+1], secIndex, 2, 0, 0, 0, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])
            uniload_info.append([0, 0, self.LateralLoad(self.bottom_chord_section)])
        return node_info, expanded_bot_index_info, ele_info, uniload_info

    def WebMemberIndexInfo(self):
        if self.truss_type.lower() == 'warren':
            if self.is_support_at_panel_point():
                topNode = self.TopIndexInfo()[:-1]
            else:
                topNode = self.TopIndexInfo()[1:-2]

            if self.bottom_chord_extension_length != 0:
                botNode = self.BotIndexInfo()[1:-1]
            else:
                botNode = self.BotIndexInfo()

            webNodeIndices = [[topNode[1], botNode[0]]]
            topNode = topNode[2:]
            for i in botNode:
                for j in range(2):
                    webNodeIndices.append([topNode[j], i])
                topNode = topNode[1:]
            webNodeIndices.append([topNode[-1], botNode[-1]])

        elif self.truss_type.lower() == 'modifiedwarren':
            if self.is_support_at_panel_point():
                topNode = self.TopIndexInfo()[1:-1]
            else:
                topNode = self.TopIndexInfo()[2:-2]

            if self.bottom_chord_extension_length != 0:
                botNode = self.BotIndexInfo()[1:-1]
            else:
                botNode = self.BotIndexInfo()
            topNode = topNode[:]
            webNodeIndices = []
            for i in botNode:
                for j in range(3):
                    webNodeIndices.append([topNode[j], i])
                topNode = topNode[2:]

        else:
            raise ValueError(f'Unknown truss_type: {self.truss_type}')
            
        return webNodeIndices

    def WebEleInfo(self):
        node_info = []
        ele_info = []
        uniload_info = []
        expanded_web_index_info = [] # @todo: make modify existing copy self.WebMemberIndexInfo

        if self.pin_web_members:
            pinned = 1
        else:
            pinned = 0

        curMax = max(self.BotChordInfo()[1])
        x_coords = self.top_chord_panel_point_x_coords() + self.bottom_chord_panel_point_x_coords()
        de = self.effective_depth()

        if self.maximum_web_member_element_length != 0 and self.maximum_web_member_element_length != None:
            for i, j in enumerate(self.WebMemberIndexInfo()):
                diff = dist([x_coords[j[1]-1], 0], [x_coords[j[0]-1], -de])
                fac = ceil(diff/self.maximum_web_member_element_length)
                if fac != 1:
                    n = j.copy()
                    x0 = x_coords[j[0]-1]
                    xf = x_coords[j[1]-1]
                    for k in list(range(fac))[1:]:
                        node_info.append([x0 + k*(xf-x0)/fac, -k*de/fac, 0])
                        n.insert(k, curMax+k)
                    expanded_web_index_info.append(n)
                    curMax = max(n)
        else:
            expanded_web_index_info = self.WebMemberIndexInfo()
        
        web_section_list = self.web_sections_all()
        for i, j in enumerate(expanded_web_index_info):
            sec = web_section_list[i]
            secIndex = self.SectionInfo()[1].index(sec) + 1
            if x_coords[j[-1]-1] > x_coords[j[0]-1]:
                beta = pi
            else:
                beta = 0
            for k, l in enumerate(j[:-1]):
                if k==0:
                    ele_info.append([j[k+1], l, secIndex, 3, beta, 0, pinned, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])
                elif k==len(j)-2:
                    ele_info.append([j[k+1], l, secIndex, 3, beta, pinned, 0, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])
                else:
                    ele_info.append([j[k+1], l, secIndex, 3, beta, 0, 0, 0, 0, inf, inf, inf, inf, inf, inf, inf, inf])                                      
                uniload_info.append([0, 0, self.LateralLoad(sec)])

        return node_info, expanded_web_index_info, ele_info, uniload_info

    # Joist Info
    def SectionInfo(self):
        sect_info=[]
        sect_name=[]
        for name, section in self.sections.items():
            sect_info.append(section.MASTAN2_sect_info())
            sect_name.append(name)
        return sect_info, sect_name
   
    def LateralLoad(self, sec):
        h = self.sections[sec].in_plane_depth()
        return self.lateral_pressure*h

    def saveMASTAN2(self, model_title=None):
        
        if model_title is None:
            model_title = self.joist_name

        node_info_top, expanded_top_index_info, ele_info_top, uniload_info_top = self.TopChordInfo()
        node_info_bot, expanded_bot_index_info, ele_info_bot, uniload_info_bot = self.BotChordInfo()
        node_info_web, expanded_web_index_info, ele_info_web, uniload_info_web = self.WebEleInfo()
        
        # Model information
        node_info = self.TopNodeInfo() + self.BotNodeInfo() + node_info_top + node_info_bot + node_info_web
        elem_info = ele_info_top + ele_info_bot + ele_info_web
        uniload_info = uniload_info_top + uniload_info_bot + uniload_info_web
        sect_info, sect_name = self.SectionInfo()
        num_top_chord_nodes = len(self.top_chord_panel_point_x_coords())

        # Support Information
        num_nodes = len(node_info)
        support_info = np.zeros([num_nodes,6])
        support_info[:] = nan
        if self.top_chord_panel_lengths[0] < self.support_location:
            support_info[2] = [0, 0, 0, 0, nan, nan]
            support_info[num_top_chord_nodes-1] = [nan, 0, 0, 0, nan, nan]   
        else:
            support_info[1] = [0, 0, 0, 0, nan, nan]
            support_info[num_top_chord_nodes-2] = [nan, 0, 0, 0, nan, nan]
        
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
        save_MASTAN2(model_title=model_title, 
                     node_info=node_info, 
                     elem_info=elem_info,
                     support_info=support_info, 
                     uniload_info=uniload_info, 
                     sect_info=sect_info,
                     sect_name=sect_name, 
                     mat_info=mat_info,
                     mat_name=mat_name)

    # Plotting Functions
    def plot_joist(self):
        node_coords = [i[0:2] for i in self.TopNodeInfo() + self.BotNodeInfo()]
        tc_x = self.top_chord_panel_point_x_coords()
        bc_x = self.bottom_chord_panel_point_x_coords()
        de = self.effective_depth()

        # Plot top chord
        plt.plot(tc_x, [0]*len(tc_x), 'k')
        plt.plot(bc_x, [-de]*len(bc_x), 'k')
        for i in node_coords:
            plt.plot(i[0], i[1], 'ko')

        # Plot web members
        for i in self.WebMemberIndexInfo():
            plt.plot([node_coords[i[0]-1][0], node_coords[i[1]-1][0]], [node_coords[i[0]-1][1], node_coords[i[1]-1][1]], 'k')
            plt.text((node_coords[i[0]-1][0]+node_coords[i[1]-1][0])/2, (node_coords[i[0]-1][1]+node_coords[i[1]-1][1])/2, 
                     self.web_sections_all()[self.WebMemberIndexInfo().index(i)], fontsize=8, ha='center', va='center'
            )

        plt.ylim([-self.depth*1.1, 0.5])
        plt.show()
