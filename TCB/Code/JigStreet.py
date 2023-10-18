#%%
# Created 7/5/2023 by Michael Tan & Josh Green
# Utilized for importing DXF shapes (curated file), generating a mask, than outputting a dxf
# Parsing DXF
import ezdxf as ez
# Maths
import numpy as np
import pandas as pd
# Shapes
import shapely.geometry as spg
from shapely.ops import unary_union, polygonize
from descartes import PolygonPatch
from shapely import affinity
# Plotting
import pylab as pl
# Misc
import os
import statistics

# %%
class loaddxf():
    # This is a class object for loading a single DXF file
    def __init__(self,fn):
        # Initialization
        self.fn = fn
        self.dbf = False #debug flag
        self.components = []
        self.cnames = []
        
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
                    print("Package Outline: ")
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
            P1X.append(round(comp[0][0],2))
            P1Y.append(round(comp[0][1],2))
            P3X.append(round(comp[2][0],2))
            P3Y.append(round(comp[2][1],2))
            P2X.append(round(comp[1][0],2))
            P2Y.append(round(comp[1][1],2))
            P4X.append(round(comp[3][0],2))
            P4Y.append(round(comp[3][1],2))
        DX = np.subtract(P3X, P1X)
        DY = np.subtract(P3Y, P1Y)
        CX = np.add(P1X,DX/2)
        CY = np.add(P1Y,DY/2)
        # One DF with centers and corners
        self.df = pd.DataFrame({"CNAME":self.cnames,"X":CX,"Y":CY,"P1X":P1X,"P1Y":P1Y,"P3X":P3X,"P3Y":P3Y})
        print("did it save?")
        self.df_full = pd.DataFrame({"CNAME":self.cnames,"X":CX,"Y":CY,"P1X":P1X,"P1Y":P1Y,
                                     "P2X":P2X,"P2Y":P2Y,
                                     "P3X":P3X,"P3Y":P3Y,
                                     "P4X":P4X,"P4Y":P4Y})
        df_m1 = pd.melt(self.df_full, id_vars=['CNAME'], value_vars=['P1X', 'P2X', "P3X", "P4X"], value_name = "X")
        df_m2 = pd.melt(self.df_full, id_vars=['CNAME'], value_vars=['P1Y', 'P2Y', "P3Y", "P4Y"], value_name = "Y")
        self.df_melt = pd.concat([df_m1,df_m2],axis=1)
        self.df_points = self.df_melt[['X','Y']].values.tolist()
        # self.dfs = pd.concat([pd.DataFrame({"X":P1X,"Y":P1Y}), pd.DataFrame({"X":P2X,"Y":P2Y}), pd.DataFrame({"X":P3X,"Y":P3Y}),
        #                         pd.DataFrame({"X":P4X,"Y":P4Y})]).reset_index(drop=True)
        # self.dfs = self.dfs.mul(self.poversize)
        print("Completed mapping to dataframes df")

    def maptolistofpoly(self):
        # This is messy for now. We take points than shove it into a box. It would be better to go from dxf vector to polygon.
        # Initialize an empty list
        self.pg_cmp = []
        for index, row in self.df.iterrows():
            p1 = (row.P1X,row.P1Y,row.P3X,row.P3Y)
            poly1 = spg.box(p1[0],p1[1],p1[2],p1[3])
            self.pg_cmp.append(poly1)
            
    def process(self):
        # Load data
        self.loaddrawing()
        # Map to dataframe
        self.maptodataframe()
        # Map to list of polygons
        self.maptolistofpoly()
        print("Completed loading all the data")
        
class generatemasks():
    # Class used to generate a mask
    def __init__(self,cmp,exp_cmp=1,exp_out=1.25):
        # Initialize some variables
        self.dbf = False #debug flag
        self.cmp = cmp
        # Variable to expand components - polygons are extended with a buffer function - magnitude
        self.exp_cmp = exp_cmp
        # Variable to expand outline - simple scaler/multiplier
        self.exp_out = exp_out
    
    def processcomp(self,expopt=False):
        # Create 3 masks: 
            # cmp_mask: component mask
            # cmp_out: component outline
            # diff_mask: Mask of the difference between outline & mask
        # Combine all smaller polygons & expand with exp_cmp
        if expopt == True:
            for i in np.arange(0.1, 0.8, 0.1):          
                pgroup = [x.buffer(i,8,2).envelope for x in self.cmp]
                cmp_mask = unary_union(pgroup)
                cmp_outline = spg.box(cmp_mask.bounds[0]-self.exp_out,cmp_mask.bounds[1]-self.exp_out,
                    cmp_mask.bounds[2]+self.exp_out,cmp_mask.bounds[3]+self.exp_out)#.buffer(1500)
                self.cmp_mask = cmp_mask
                self.cmp_out = cmp_outline
                # Create a difference mask between outline & smaller polygons
                self.diff_mask = cmp_outline.difference(cmp_mask)
                parea = [x.area for x in self.diff_mask]
                print("Loop at " + str(i))
                print(str(statistics.median(parea)))
                if statistics.median(parea) >=1:
                    print("Minimum small features detected. Buffer = "+str(i))
                    print(str(parea))
                    self.diff_mask = spg.MultiPolygon(list(filter(lambda x: x.area> 1, self.diff_mask)))
                    break

                
        else:
            pgroup = [x.buffer(self.exp_cmp).envelope.simplify(1) for x in self.cmp]
            cmp_mask = unary_union(pgroup)
            # Create an outline using the cmp_mask X&Y Min/Max values
            cmp_outline = spg.box(cmp_mask.bounds[0]-self.exp_out,cmp_mask.bounds[1]-self.exp_out,
                            cmp_mask.bounds[2]+self.exp_out,cmp_mask.bounds[3]+self.exp_out)#.buffer(1500)
            self.cmp_mask = cmp_mask
            self.cmp_out = cmp_outline
            # Create a difference mask between outline & smaller polygons
            self.diff_mask = cmp_outline.difference(cmp_mask)
            print("created component mask with: Expand = " + str(self.exp_cmp) + " and " + str(self.exp_out) + "x outline.")

        
    def process(self):
        print("Running mask script")
        self.processcomp()
        print("Completed")
               
class plotfun():
    # Function to make plotting just a little easier as I am lazy
    def __init__(self,pg_cmp,diff_poly,fs=(20,10)):
        self.dbf = False #debug flag
        self.fs = fs
        self.pg_cmp = pg_cmp
        self.diff_poly = diff_poly
        
    # def plot_polygon(self,data):
    #     # Simple Function to plot polygons
    #     fig = pl.figure(figsize=self.fs)
    #     ax = fig.add_subplot(111)
    #     margin = 1
    #     x_min, y_min, x_max, y_max = data.bounds
    #     ax.set_xlim([x_min-margin, x_max+margin])
    #     ax.set_ylim([y_min-margin, y_max+margin])
    #     patch = PolygonPatch(data, fc='#999999', ec='#000000', fill=True, zorder=-1)
    #     ax.add_patch(patch)
    #     fig = pl.plot(self.x,self.y,'o', color='#f16824')
    #     return fig
    
    # def plot_points(self):
    #     # Simple function to plot points ()
    #     fig = pl.figure(figsize=self.fs)
    #     fig = pl.plot(self.x,self.y,'o', color='#f16824')
    #     return fig
    
    def plot_results(self):
        # Simple function to plot everything together - components & difference mask
        fig = pl.figure(figsize=self.fs)
        ax = fig.add_subplot(111)
        margin = 1
        x_min, y_min, x_max, y_max = self.diff_poly.bounds
        ax.set_xlim([x_min-margin, x_max+margin])
        ax.set_ylim([y_min-margin, y_max+margin])
        patch1 = PolygonPatch(self.diff_poly, fc='green', ec='#000000', fill=True, zorder=-1)
        patch2 = PolygonPatch(unary_union(self.pg_cmp), fc='red', ec='#000000', fill=True, zorder=-1)
        ax.add_patch(patch1)
        ax.add_patch(patch2)
        ax.set_title('Masky mask doing mask things')
        # return fig
    
    # Temp
    # boundary = gpd.GeoSeries(cmp_mask)
    # outline = gpd.GeoSeries(cmp_outline)
    # diff = gpd.GeoSeries(diff_poly)
    
class generatedxf():
    def __init__(self,fn,cmp_mask,diff_mask):
        self.outfile = os.path.splitext(fn)[0]+"_jigstreet.dxf"
        self.cmp_mask = cmp_mask
        self.diff_mask = diff_mask
    
    def addoutline(self):
        for thispoly in self.cmp_mask:
            dt = pd.DataFrame({"X":thispoly.exterior.coords.xy[0], "Y":thispoly.exterior.coords.xy[1]})
            dt['xy'] = dt.apply(lambda x: (x['X'], x['Y']), axis=1)
            self.msp.add_lwpolyline(dt['xy'].tolist())
    
    def addholes(self):
        for thispoly in self.diff_mask:
            dt = pd.DataFrame({"X":thispoly.exterior.coords.xy[0], "Y":thispoly.exterior.coords.xy[1]})
            dt['xy'] = dt.apply(lambda x: (x['X'], x['Y']), axis=1)
            self.msp.add_lwpolyline(dt['xy'].tolist())

    def process(self):
        self.doc = ez.readfile(fn)
        self.msp = self.doc.modelspace()
        self.doc.layers.new("JIG_STREET")
        self.addoutline()
        self.addholes()
        self.doc.saveas(self.outfile)
        print("Saved "+ str(self.outfile))   
#%%
if __name__ == "__main__":
    # Main script to run
    fn = r"E:\Scripting\MACD\MACD\TCB\Examples\M54481-001_BSR_r01-Scaled.dxf"
    # fn = r"E:\Scripting\dxf\M86710-001_BSR_Cleaned Up.dxf"
    # fn = r"E:\Scripting\dxf\input_example.dxf"
    derp = loaddxf(fn)
    derp.process()
    masks = generatemasks(derp.pg_cmp,.75,1)
    masks.process()
    meow = plotfun(derp.pg_cmp,masks.diff_mask,(10,20))
    meow.plot_results()
    # %%
    woof = generatedxf(fn,masks.cmp_mask,masks.diff_mask)
    woof.process()

 # %%