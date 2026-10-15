import numpy as np
from dataclasses import dataclass
from typing import Optional
from math import inf, pi, sin, cos, sqrt
from libdenavit.section import DoubleAngle, Angle
from shapely import Polygon
from sectionproperties.pre import Geometry
from sectionproperties.analysis import Section


def mastan2_sect_info(*, A, Izz, Iyy, J, Cw, Zzz, Zyy, Ayy, Azz,
                      yield_surface_P=1, yield_surface_Mz=1, yield_surface_My=1,
                      is_symmetric=1, Ysc=0, Zsc=0,
                      beta_y=0, beta_z=0, beta_w=0, phi=0, Iyz=0):
    """Assemble the 20-column section row that MASTAN2 expects.

    Column order is defined by ``libdenavit.MASTAN2.save_MASTAN2``:

    ===  ===============================  ===  ===============================
      1  Area                              11  Yield surface factor for Mz
      2  Moment of inertia Izz             12  Yield surface factor for My
      3  Moment of inertia Iyy             13  Is symmetric? 1 = yes, 0 = no
      4  Torsion constant J                14  Shear center Ysc
      5  Warping coefficient Cw            15  Shear center Zsc
      6  Plastic section modulus Zzz       16  Wagner beta_y
      7  Plastic section modulus Zyy       17  Wagner beta_z
      8  Shear area Ayy                    18  Wagner beta_w
      9  Shear area Azz                    19  Phi, radians
     10  Yield surface factor for P        20  Product of inertia Iyz
    ===  ===============================  ===  ===============================

    MASTAN2's local z axis is the in-plane bending axis for a frame modelled in
    the global x-y plane, so ``Izz`` takes the in-plane moment of inertia.
    """
    return [A, Izz, Iyy, J, Cw, Zzz, Zyy, Ayy, Azz,
            yield_surface_P, yield_surface_Mz, yield_surface_My, is_symmetric,
            Ysc, Zsc, beta_y, beta_z, beta_w, phi, Iyz]


@dataclass(frozen=True)
class Round:
    D: float
    
    A: Optional[float] = None
    I: Optional[float] = None
    J: Optional[float] = None
    Z: Optional[float] = None
    As: Optional[float] = None
    r: Optional[float] = None

    def __post_init__(self):       
        if self.A is None:
            object.__setattr__(self, "A", pi/4*self.D**2)
        if self.I is None:
            object.__setattr__(self, "I", pi/64*self.D**4)
        if self.J is None:
            object.__setattr__(self, "J", pi/32*self.D**4)
        if self.Z is None:
            object.__setattr__(self, "Z", 1/6*self.D**3)
        if self.As is None:
            object.__setattr__(self, "As", 0.9*self.A)
        if self.r is None:
            object.__setattr__(self, "r", self.D/4)

    def MASTAN2_sect_info(self):
        return mastan2_sect_info(
            A=self.A, Izz=self.I, Iyy=self.I,
            J=self.J, Cw=0, Zzz=self.Z, Zyy=self.Z,
            Ayy=self.As, Azz=self.As)

    def in_plane_depth(self):
        return self.D

    def slenderness(self,L):
        return L/self.r


@dataclass(frozen=True)
class HotRolledDoubleAngle:
    b: float
    t: float
    s: float
    d: Optional[float] = None
    
    A: Optional[float] = None
    y_bar: Optional[float] = None
    Ix: Optional[float] = None
    Iy: Optional[float] = None
    J: Optional[float] = None
    Zx: Optional[float] = None
    Zy: Optional[float] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    Asx: Optional[float] = None
    Asy: Optional[float] = None
    rz_single: Optional[float] = None
    
    number_of_fillers: Optional[int] = 0

    def __post_init__(self):
        if self.d is None:
            object.__setattr__(self, "d", self.b)
        
        obj = DoubleAngle(self.d, self.b, self.t, self.s)
        if self.A is None:
            object.__setattr__(self, "A", obj.A)
        if self.y_bar is None:
            object.__setattr__(self, "y_bar", obj.y_bar)
        if self.Ix is None:
            object.__setattr__(self, "Ix", obj.Ix) 
        if self.Iy is None:
            object.__setattr__(self, "Iy", obj.Iy)
        if self.J is None:
            object.__setattr__(self, "J", obj.J)
        if self.Zx is None:
            object.__setattr__(self, "Zx", obj.Zx)
        if self.Zy is None:
            object.__setattr__(self, "Zy", obj.Zy)
        if self.rx is None:
            object.__setattr__(self, "rx", obj.rx)
        if self.ry is None:
            object.__setattr__(self, "ry", obj.ry)
        if self.Asx is None:
            object.__setattr__(self, "Asx", 2*self.d*self.t)
        if self.Asy is None:
            object.__setattr__(self, "Asy", 2*self.b*self.t)
            
        obj = Angle(self.d, self.b, self.t)
        if self.rz_single is None:
            object.__setattr__(self, "rz_single", obj.rz)           

    def MASTAN2_sect_info(self):
        return mastan2_sect_info(
            A=self.A, Izz=self.Ix, Iyy=self.Iy, 
            J=self.J, Cw=0,
            Zzz=self.Zx, Zyy=self.Zy,
            Ayy=self.Asx, Azz=self.Asy)

    def in_plane_depth(self):
        return self.d
        
    def slenderness(self,L):
        l_over_rx = L/self.rx
        l_over_ry = L/self.ry
        l_over_rz = L/(self.number_of_fillers+1)/self.rz_single
        return max(l_over_rx,l_over_ry,l_over_rz)


@dataclass(frozen=True)
class HotRolledCrimpedAngle:
    # Assume always equal leg angle
    b: float
    t: float

    A: Optional[float] = None
    Ix: Optional[float] = None
    Iy: Optional[float] = None
    J: Optional[float] = None
    yp: Optional[float] = None
    Zx: Optional[float] = None
    Zy: Optional[float] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    Asx: Optional[float] = None
    Asy: Optional[float] = None

    def __post_init__(self):
        obj = Angle(self.b, self.b, self.t)
        if self.A is None:
            object.__setattr__(self, "A", obj.A)
        if self.Ix is None:
            object.__setattr__(self, "Ix", (obj.Ix + obj.Iy)/2 + obj.Ixy) 
        if self.Iy is None:
            object.__setattr__(self, "Iy", (obj.Ix + obj.Iy)/2 - obj.Ixy)
        if self.J is None:
            object.__setattr__(self, "J", obj.J)
        if self.yp is None:
            object.__setattr__(self, "yp", (self.b+1.5*self.t)/(2*sqrt(2)))
        if self.Zx is None:
            object.__setattr__(self, "Zx", (self.yp**3 - (self.yp-sqrt(2)*self.t)**3)/3 + self.t**3/(3*sqrt(2)) + (self.b-sqrt(2)*self.yp)*self.t*(self.b-sqrt(2)*self.yp+self.t)/sqrt(2))
        if self.Zy is None:
            object.__setattr__(self, "Zy", (self.t**3/3 + self.b*self.t*(self.b-self.t))/sqrt(2))
        if self.rx is None:
            object.__setattr__(self, "rx", sqrt(self.Ix/self.A))
        if self.ry is None:
            object.__setattr__(self, "ry", sqrt(self.Iy/self.A))
        if self.Asx is None:
            object.__setattr__(self, "Asx", (5/6)*self.b*self.t)
        if self.Asy is None:
            object.__setattr__(self, "Asy", (5/6)*self.b*self.t)

    def cross_section_points(self):
        root2 = sqrt(2)
        pts = []
        pts.append((0, 0))
        pts.append((self.b/root2, -self.b/root2))
        pts.append(((self.b-self.t)/root2, -(self.b+self.t)/root2))
        pts.append((0, -2*self.t/root2))
        pts.append((-(self.b-self.t)/root2, -(self.b+self.t)/root2))
        pts.append((-self.b/root2, -self.b/root2))
        return pts

    def MASTAN2_sect_info(self):
        return mastan2_sect_info(
            A=self.A, Izz=self.Ix, Iyy=self.Iy, 
            J=self.J, Cw=0,
            Zzz=self.Zx, Zyy=self.Zy,
            Ayy=self.Asx, Azz=self.Asy)

    def in_plane_depth(self):
        return (self.b+self.t)/sqrt(2)

    def slenderness(self,L):
        return L/self.rx


@dataclass(frozen=True)
class ColdFormedAngle:
    b: float
    t: float
    ri: float
    
    A: Optional[float] = None
    x_bar: Optional[float] = None
    y_bar: Optional[float] = None
    Ix: Optional[float] = None
    Iy: Optional[float] = None
    Iz: Optional[float] = None
    J: Optional[float] = None
    Cw: Optional[float] = None
    Zx: Optional[float] = None
    Zy: Optional[float] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    rz: Optional[float] = None
    Asx: Optional[float] = None
    Asy: Optional[float] = None

    num_facets: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "ro", self.ri+self.t)
        
        if self.num_facets is None:
            object.__setattr__(self, "num_facets", 20)
        
        geom = Geometry(geom=Polygon(self.cross_section_points()))
        #geom.plot_geometry()
        geom.create_mesh(mesh_sizes=0.01*(self.t)**2)
        sec = Section(geometry=geom)
        #sec.plot_mesh()
        sec.calculate_geometric_properties()
        sec.calculate_warping_properties()
        sec.calculate_plastic_properties()
        #sec.display_results()
    
        if self.A is None:
            object.__setattr__(self, "A", float(sec.get_area()))
        if self.x_bar is None:
            object.__setattr__(self, "x_bar", float(-sec.get_c()[0]))
        if self.y_bar is None:
            object.__setattr__(self, "y_bar", float(-sec.get_c()[1]))
        if self.Ix is None:
            object.__setattr__(self, "Ix", float(sec.get_ic()[0]))
        if self.Iy is None:
            object.__setattr__(self, "Iy", float(sec.get_ic()[1]))
        if self.Iz is None:
            object.__setattr__(self, "Iz", float(sec.get_ip()[1]))
        if self.J is None:
            object.__setattr__(self, "J", float(sec.get_j()))
        if self.Cw is None:
            object.__setattr__(self, "Cw", float(sec.get_gamma()))
        if self.Zx is None:
            object.__setattr__(self, "Zx", float(sec.get_s()[0]))
        if self.Zy is None:
            object.__setattr__(self, "Zy", float(sec.get_s()[1]))
        if self.rx is None:
            object.__setattr__(self, "rx", sqrt(self.Ix/self.A)) 
        if self.ry is None:
            object.__setattr__(self, "ry", sqrt(self.Iy/self.A))
        if self.rz is None:
            object.__setattr__(self, "rz", sqrt(self.Iz/self.A))           
        if self.Asx is None:
            object.__setattr__(self, "Asx", float(sec.get_as()[0]))
        if self.Asy is None:
            object.__setattr__(self, "Asy", float(sec.get_as()[1]))

    def cross_section_points(self):
        angles = np.linspace(0,pi/2,self.num_facets+1)
        pts = []
        pts.append((0, -self.b))
        for angle in angles:
            xo = self.ro
            yo = -self.ro
            pts.append((xo + self.ro * cos(pi-angle), yo + self.ro * sin(pi-angle)))
        pts.append((self.b, 0))
        pts.append((self.b, -self.t))
        for angle in angles:
            xo = self.ro
            yo = -self.ro
            pts.append((xo + self.ri * cos(angle+pi/2), yo + self.ri * sin(angle+pi/2)))
        pts.append((self.t, -self.b))
        return pts
    
    def MASTAN2_sect_info(self):
        return mastan2_sect_info(
            A=self.A, Izz=self.Ix, Iyy=self.Iy,
            J=self.J, Cw=self.Cw,
            Zzz=self.Zx, Zyy=self.Zy, Ayy=self.Asy, Azz=self.Asz)

    def in_plane_depth(self):
        return self.h


@dataclass(frozen=True)
class ColdFormedDoubleAngle:
    # Assume always equal leg angle
    b: float
    t: float
    ri: float
    s: float
    
    A: Optional[float] = None
    y_bar: Optional[float] = None
    Ix: Optional[float] = None
    Iy: Optional[float] = None
    Iz: Optional[float] = None
    J: Optional[float] = None
    Cw: Optional[float] = None
    Zx: Optional[float] = None
    Zy: Optional[float] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    Asx: Optional[float] = None
    Asy: Optional[float] = None
    rz_single: Optional[float] = None

    num_facets: Optional[int] = None


    def __post_init__(self):
        object.__setattr__(self, "ro", self.ri+self.t)
        
        if self.num_facets is None:
            object.__setattr__(self, "num_facets", 20)
        
        single_angle = ColdFormedAngle(self.b, self.t, self.ri)
        if self.A is None:
            object.__setattr__(self, "A", 2*single_angle.A)
        if self.y_bar is None:
            object.__setattr__(self, "y_bar", single_angle.y_bar)
        if self.Ix is None:
            object.__setattr__(self, "Ix", 2*single_angle.Ix)
        if self.Iy is None:
            object.__setattr__(self, "Iy", 2*(single_angle.Iy + single_angle.A*(self.s/2 + single_angle.x_bar)**2))
        if self.J is None:
            object.__setattr__(self, "J", 2*single_angle.J)
        if self.Cw is None:
            object.__setattr__(self, "Cw", 2*single_angle.Cw)
        if self.Zx is None:
            object.__setattr__(self, "Zx", inf) # @todo - determine real value.
        if self.Zy is None:
            object.__setattr__(self, "Zy", inf) # @todo - determine real value.
        if self.rx is None:
            object.__setattr__(self, "rx", sqrt(self.Ix/self.A))
        if self.ry is None:
            object.__setattr__(self, "ry", sqrt(self.Iy/self.A))
        if self.Asx is None:
            object.__setattr__(self, "Asx", 2*single_angle.Asx)
        if self.Asy is None:
            object.__setattr__(self, "Asy", 2*single_angle.Asx)
        if self.rz_single is None:
            object.__setattr__(self, "rz_single", single_angle.rz)

    def MASTAN2_sect_info(self):
        return mastan2_sect_info(
            A=self.A, Izz=self.Ix, Iyy=self.Iy,
            J=self.J, Cw=self.Cw,
            Zzz=self.Zx, Zyy=self.Zy, Ayy=self.Asy, Azz=self.Asx)

    def in_plane_depth(self):
        return self.b


@dataclass(frozen=True)
class ColdFormedChannel:
    w: float
    h: float
    t: float
    ri: float
    
    A: Optional[float] = None
    Ix: Optional[float] = None
    Iy: Optional[float] = None
    J: Optional[float] = None
    Cw: Optional[float] = None
    Zx: Optional[float] = None
    Zy: Optional[float] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    Asx: Optional[float] = None
    Asy: Optional[float] = None

    num_facets: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "ro", self.ri+self.t)
        
        if self.num_facets is None:
            object.__setattr__(self, "num_facets", 20)

        geom = Geometry(geom=Polygon(self.cross_section_points()))
        #geom.plot_geometry()
        geom.create_mesh(mesh_sizes=0.01*(self.t)**2)
        sec = Section(geometry=geom)
        #sec.plot_mesh()
        sec.calculate_geometric_properties()
        sec.calculate_warping_properties()
        sec.calculate_plastic_properties()
        #sec.display_results()
        
        if self.A is None:
            object.__setattr__(self, "A", float(sec.get_area()))
        if self.Ix is None:
            object.__setattr__(self, "Ix", float(sec.get_ic()[0]))
        if self.Iy is None:
            object.__setattr__(self, "Iy", float(sec.get_ic()[1]))
        if self.J is None:
            object.__setattr__(self, "J", float(sec.get_j()))
        if self.Cw is None:
            object.__setattr__(self, "Cw", float(sec.get_gamma()))
        if self.Zx is None:
            object.__setattr__(self, "Zx", float(sec.get_s()[0]))
        if self.Zy is None:
            object.__setattr__(self, "Zy", float(sec.get_s()[1]))
        if self.rx is None:
            object.__setattr__(self, "rx", sqrt(self.Ix/self.A)) 
        if self.ry is None:
            object.__setattr__(self, "ry", sqrt(self.Iy/self.A))           
        if self.Asx is None:
            object.__setattr__(self, "Asx", float(sec.get_as()[0]))
        if self.Asy is None:
            object.__setattr__(self, "Asy", float(sec.get_as()[1]))

    def cross_section_points(self):
        angles = np.linspace(0,pi/2,self.num_facets+1)
        pts = []
        pts.append((-self.w / 2, 0))
        pts.append((-self.w / 2 + self.t, 0))
        for angle in angles:
            xo = -self.w / 2 + self.ro
            yo = -self.h + self.ro
            pts.append((xo + self.ri * cos(angle+pi), yo + self.ri * sin(angle+pi)))
        for angle in angles:
            xo = self.w / 2 - self.ro
            yo = -self.h + self.ro
            pts.append((xo + self.ri * cos(angle+3*pi/2), yo + self.ri * sin(angle+3*pi/2)))
        pts.append((self.w / 2 - self.t, 0))
        pts.append((self.w / 2, 0))
        for angle in angles:
            xo = self.w / 2 - self.ro
            yo = -self.h + self.ro
            pts.append((xo + self.ro * cos(-angle), yo + self.ro * sin(-angle)))
        for angle in angles:
            xo = -self.w / 2 + self.ro
            yo = -self.h + self.ro
            pts.append((xo + self.ro * cos(-angle-pi/2), yo + self.ro * sin(angle-pi/2)))
        return pts
    
    def MASTAN2_sect_info(self):
        return mastan2_sect_info(
            A=self.A, Izz=self.Ix, Iyy=self.Iy,
            J=self.J, Cw=self.Cw,
            Zzz=self.Zx, Zyy=self.Zy, Ayy=self.Asy, Azz=self.Asx)

    def in_plane_depth(self):
        return self.h
