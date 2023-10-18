#%%
# Updated 10/12/2023 by Michael Tan
# Utilized for importing DXF shapes (curated file), generating a mask, than outputting a dxf
# Parsing DXF
import ezdxf as ez
# Maths
import numpy as np
import pandas as pd
# Shapes
import shapely.geometry as spg
from shapely.ops import unary_union, polygonize
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category
import pandas
from descartes import PolygonPatch
from shapely import affinity
# Plotting
import pylab as pl
import matplotlib.pyplot as plt

# Misc
import os
import statistics

# %%
class loaddxf():
    # This is a class object for loading a single DXF file
    def __init__(self,fn,scaledrawing=1):
        # Initialization
        self.fn = fn
        self.dbf = False #debug flag
        self.components = []
        self.cnames = []
        self.scaler = scaledrawing
        
    def loaddrawing(self):
        # Import DXF as a ezdxf object. 
        doc = ez.readfile(self.fn)
        msp = doc.modelspace()
        # Iterate through all items
        for e in msp:
            if self.dbf == True:
                print(e.dxf.layer)
                print(e.dxftype())
            # If we encounter "PKG_OUTLINE" layer, export it to self.pkgout
            if e.dxf.layer == 'PKG_OUTLINE':
                if e.dxftype() == 'LWPOLYLINE':
                    #print(e.vertices(),e.dxftype())
                    p = e.get_points()
                    # update later to add polygon
                    self.pkgout = spg.box(min(p)[0],min(p)[1],max(p)[0],max(p)[1])
                    #print("Package Outline: ")
                    self.pkgout
            # If we encounter Soldermask_pads_btm, export SOLID objects to a list of components
            elif e.dxf.layer == 'SOLDERMASK_PADS_BTM':
                if e.dxftype() == 'SOLID':
                    if self.dbf == True:
                        print(e.dxf.handle)
                        print(e.vertices())
                    self.cnames.append(e.dxf.handle)
                    self.components.append(e.vertices())
        print(str(len(self.components)) + " objects have been found on BSR layer.")
        
    def maptodataframe(self):
        # Using the solid objects encountered, parse into a dataframe or 2.
        P1X,P1Y = [],[]
        P2X,P2Y = [],[]
        P3X,P3Y = [],[]
        P4X,P4Y = [],[]
        CX,CY = [],[]
        for comp in self.components:
            # Identify point# 1 - bottom left corner
            P1X.append(comp[0][0]*self.scaler)
            P1Y.append(comp[0][1]*self.scaler)
            P3X.append(comp[2][0]*self.scaler)
            P3Y.append(comp[2][1]*self.scaler)
            P2X.append(comp[1][0]*self.scaler)
            P2Y.append(comp[1][1]*self.scaler)
            P4X.append(comp[3][0]*self.scaler)
            P4Y.append(comp[3][1]*self.scaler)
        # Sizes
        DX = np.subtract(P3X, P1X)
        DY = np.subtract(P3Y, P1Y)
        # Centers
        CX = np.add(P1X,DX/2)
        CY = np.add(P1Y,DY/2)
        # One DF with centers and corners
        self.df = pd.DataFrame({"CNAME":self.cnames,"X":CX,"Y":CY,"P1X":P1X,"P1Y":P1Y,"P3X":P3X,"P3Y":P3Y})
        self.df_full = pd.DataFrame({"CNAME":self.cnames,"X":CX,"Y":CY,"P1X":P1X,"P1Y":P1Y,
                                     "P2X":P2X,"P2Y":P2Y,
                                     "P3X":P3X,"P3Y":P3Y,
                                     "P4X":P4X,"P4Y":P4Y})
        df_m1 = pd.melt(self.df_full, id_vars=['CNAME'], value_vars=['P1X', 'P2X', "P3X", "P4X"], value_name = "X")
        df_m2 = pd.melt(self.df_full, id_vars=['CNAME'], value_vars=['P1Y', 'P2Y', "P3Y", "P4Y"], value_name = "Y")
        self.df_melt = pd.concat([df_m1,df_m2],axis=1)
        self.df_points = self.df_melt[['X','Y']].values.tolist()
        print("Completed mapping to dataframes df (center, size, corner points)")

    def maptolistofpoly(self):
        # This is messy for now. We take points than shove it into a box. It would be better to go from dxf vector to polygon.
        # Initialize an empty list
        self.pg_cmp = []
        for index, row in self.df.iterrows():
            p1 = (row.P1X,row.P1Y,row.P3X,row.P3Y)
            poly1 = spg.box(p1[0],p1[1],p1[2],p1[3])
            self.pg_cmp.append(poly1)
        print("pg_cmp is nominal position of all package components.")
            
    def process(self):
        # Load data
        self.loaddrawing()
        print("loaded raw drawing")
        # Map to dataframe
        self.maptodataframe()
        print("Mapped to dataframes")
        # Map to list of BOXES - no oversize. 
        self.maptolistofpoly()
        print("mapped to polygons")
        print("Completed loading all the data")
        
class generatemasks():
    # Class used to generate a mask
    def __init__(self,cmp,exp_cmp=0.65,exp_out=1):
        # Initialize some variables
        self.dbf = False #debug flag
        self.cmp = cmp
        # Variable to expand components - polygons are extended with a buffer function - magnitude
        self.exp_cmp = exp_cmp
        # Variable to expand outline - simple scaler/multiplier
        self.exp_out = exp_out
    
    def process_cmp_outline(self):
        # Create 3 masks: 
            # cmp_mask: component mask
            # cmp_out: component outline
            # diff_mask: Mask of the difference between outline & mask
        # Combine all smaller polygons & expand with exp_cmp
        bff_cmp = [x.buffer(self.exp_cmp).envelope.buffer(1).buffer(-1) for x in self.cmp]
        self.bff_cmp_mask = unary_union(bff_cmp)
        self.raw_cmp_mask = unary_union(self.cmp)
        print("generated outline")
    
    def process_cmp_interior(self):
        
        cmp_outline_mask = spg.box(self.bff_cmp_mask.bounds[0]-self.exp_out,self.bff_cmp_mask.bounds[1]-self.exp_out,
            self.bff_cmp_mask.bounds[2]+self.exp_out,self.bff_cmp_mask.bounds[3]+self.exp_out)#.buffer(1500)
        diff_mask = cmp_outline_mask.difference(self.bff_cmp_mask)
        
        if diff_mask.geom_type == 'MultiPolygon':
            self.interior_sup_mask = spg.MultiPolygon([P.buffer(1).buffer(-1).simplify(0.05) for P in diff_mask.geoms if P.area < 50])
            print("generated interior")
        elif diff_mask.geom_type == 'Polygon':
            self.interior_sup_mask = False
            print("No interior supports with single polygon")

    def process(self):
        print("Running mask script")
        self.process_cmp_outline()
        self.process_cmp_interior()
        print("created component mask with: Expand = " + str(self.exp_cmp) + " and " + str(self.exp_out) + "x outline.")
        print("cmp_mask is raw mask, cmp_out is outline mask, diff_mask is outline-component")
        print("Completed")

class plotfun():
    
    def __init__(self,pkgout,raw_cmp_mask, bff_cmp_mask, interior_sup_mask):
        # Initialize some variables
        self.dbf = False #debug flag
        self.raw_cmp_mask = raw_cmp_mask
        self.bff_cmp_mask = bff_cmp_mask
        self.interior_sup_mask = interior_sup_mask
        self.pkgout = pkgout
    
    def plotstuff(self):
        for geom in self.raw_cmp_mask.geoms:
            plt.plot(*geom.exterior.xy, color='gray')
        if type(self.raw_cmp_mask) == type(self.bff_cmp_mask):
            for geom in self.bff_cmp_mask.geoms:
                plt.plot(*geom.exterior.xy)
        else:
            plt.plot(*self.bff_cmp_mask.exterior.xy)
        if self.interior_sup_mask is not False:
            for geom in self.interior_sup_mask.geoms:
                plt.plot(*geom.exterior.xy)
        plt.plot(*self.pkgout.exterior.xy)
        # Set (current) axis to be equal before showing plot
        plt.gca().axis("equal")
        plt.show()
    
    def process(self):
        self.plotstuff()

def processjigstreet(fn):
    print("loading a single file")
    root = r"E:\Scripting\MACD\MACD\TCB"
    dxf = fn
    dxffile = os.path.join(root,"Examples",dxf)
    print("File to process: " + dxffile)
    rawdxf = loaddxf(dxffile,scaledrawing=1)
    rawdxf.process()
    print(rawdxf.pkgout)
    print("Generated pg_cmp, df, df_melt, and df_points")
    maskset = generatemasks(rawdxf.pg_cmp)
    maskset.process()
    plotme = plotfun(rawdxf.pkgout,maskset.raw_cmp_mask, maskset.bff_cmp_mask, maskset.interior_sup_mask)
    plotme.process()
# %%
root = r"E:\Scripting\MACD\MACD\TCB"
dxfs = []
for file in os.listdir(os.path.join(root,"Examples")):
    if file.endswith(".dxf"):
        dxfs.append(os.path.join(root,"Examples",file))
print(dxfs)
for dxf in dxfs:
    try:
        processjigstreet(dxf)
    except:
        print("\n" + dxf + " had some issues ... moving on" + "\n")
# derp = processjigstreet("M54481-001_BSR_r01-Scaled.dxf")

# %%
