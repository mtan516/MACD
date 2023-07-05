#%%
import ezdxf as ez
import sys
import numpy as np
import pandas as pd
import shapely.geometry as spg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shapely as sp
import geopandas as gpd
import pylab as pl
from descartes import PolygonPatch
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import math
from matplotlib.collections import LineCollection


# %%
class loaddxf():
    def __init__(self,fn,poversize=1):
        self.fn = fn
        self.dbf = False #debug flag
        self.components = []
        self.cnames = []
        self.poversize = poversize
        
    def loaddrawing(self):
        doc = ez.readfile(self.fn)
        msp = doc.modelspace()
        for e in msp:
            if self.dbf == True:
                print(e.dxf.layer)
                print(e.dxftype())
                
            if e.dxf.layer == 'PKG_OUTLINE':
                if e.dxftype() == 'LWPOLYLINE':
                    #print(e.vertices(),e.dxftype())
                    p = e.get_points()
                    # update later to add polygon
                    self.pkgout = spg.box(min(p)[0],min(p)[1],max(p)[0],max(p)[1])
                    print("Package Outline: ")
                    self.pkgout
            
            elif e.dxf.layer == 'SOLDERMASK_PADS_BTM':
                if e.dxftype() == 'SOLID':
                    if self.dbf == True:
                        print(e.dxf.handle)
                        print(e.vertices())
                    self.cnames.append(e.dxf.handle)
                    self.components.append(e.vertices())
        print(str(len(self.components)) + " objects have been found on BSR layer.")
        
    def maptodataframe(self):
        P1X,P1Y = [],[]
        P2X,P2Y = [],[]
        P3X,P3Y = [],[]
        P4X,P4Y = [],[]
        CX,CY = [],[]
        for comp in self.components:
            # Identify point# 1 - bottom left corner
            P1X.append(round(comp[0][0],0))
            P1Y.append(round(comp[0][1],0))
            P3X.append(round(comp[2][0],0))
            P3Y.append(round(comp[2][1],0))
            P2X.append(round(comp[1][0],0))
            P2Y.append(round(comp[1][1],0))
            P4X.append(round(comp[3][0],0))
            P4Y.append(round(comp[3][1],0))
        DX = np.subtract(P3X, P1X)
        DY = np.subtract(P3Y, P1Y)
        CX = np.add(P1X,DX/2)
        CY = np.add(P1Y,DY/2)
        # One DF with centers and corners
        self.df = pd.DataFrame({"CNAME":self.cnames,"X":CX,"Y":CY,"P1X":P1X,"P1Y":P1Y,"P3X":P3X,"P3Y":P3Y})
        self.dfs = pd.concat([pd.DataFrame({"X":P1X,"Y":P1Y}), pd.DataFrame({"X":P2X,"Y":P2Y}), pd.DataFrame({"X":P3X,"Y":P3Y}),
                                pd.DataFrame({"X":P4X,"Y":P4Y})]).reset_index(drop=True)
        self.dfs = self.dfs.mul(self.poversize)
        print("Completed mapping to dataframes df & dfs")

    def cmpoutline(self):
        self.pg_cmp = []
        dt = self.df
        for index, row in dt.iterrows():
            # print(row.P1X)
            p1 = (row.P1X,row.P1Y,row.P3X,row.P3Y)
            # print(p1)
            poly1 = spg.box(p1[0],p1[1],p1[2],p1[3])
            self.pg_cmp.append(poly1)
            
    def process(self):
        self.loaddrawing()
        self.maptodataframe()
        self.cmpoutline()
        print("Completed")
        
class generatemasks():
    
    def __init__(self,cmp,exp_cmp=500,exp_out=1.25):
        self.df = df
        self.dbf = False #debug flag
        self.cmp = cmp
        self.exp_cmp = exp_cmp
        self.exp_out = exp_out
    
    def processcomp(self):
        expand_n = 500
        scaler = 1.25
        cmp_mask = unary_union([x.buffer(self.exp_cmp) for x in self.cmp])
        cmp_outline = spg.box(cmp_mask.bounds[0]*self.exp_out,cmp_mask.bounds[1]*self.exp_out,
                        cmp_mask.bounds[2]*self.exp_out,cmp_mask.bounds[3]*self.exp_out)#.buffer(1500)
        self.cmp_mask = cmp_mask
        self.cmp_out = cmp_outline
        self.diff_mask = cmp_outline.difference(cmp_mask)
        print("created component mask with: Expand = " + str(expand_n) + " and " + str(scaler) + "x outline.")

        
    def process(self):
        print("Running mask script")
        self.processcomp()
        print("Completed")
        
    # boundary = gpd.GeoSeries(cmp_mask)
    # outline = gpd.GeoSeries(cmp_outline)
    # diff = gpd.GeoSeries(diff_poly)
    
                
class generatepointmask():
    def __init__(self,df,fs=(20,10)):
        self.df = df
        self.dbf = False #debug flag
        self.fs = fs
        
    def maptopoints(self):
        self.points = gpd.GeoDataFrame(self.df, geometry=gpd.points_from_xy(self.df.X, self.df.Y)).reset_index(drop=True)
        self.points = self.points.geometry
        if self.dbf == True:
            print(self.points)
        self.x = [p.coords.xy[0] for p in self.points]
        self.y = [p.coords.xy[1] for p in self.points]
        self.point_collection = sp.geometry.MultiPoint(list(self.points))
        self.point_collection.envelope
    
    # def simpleconvexhull(self):
    #     self.convex_hull_polygon = self.point_collection.convex_hull
        
    # def genalpha(self, alpha):
    #     points = self.points
    #     """
    #     Compute the alpha shape (concave hull) of a set of points.

    #     @param points: Iterable container of points.
    #     @param alpha: alpha value to influence the gooeyness of the border. Smaller
    #                 numbers don't fall inward as much as larger numbers. Too large,
    #                 and you lose everything!
    #     """
    #     if len(points) < 4:
    #         # When you have a triangle, there is no sense in computing an alpha
    #         # shape.
    #         return sp.geometry.MultiPoint(list(points)).convex_hull

    #     def add_edge(edges, edge_points, coords, i, j):
    #         """Add a line between the i-th and j-th points, if not in the list already"""
    #         if (i, j) in edges or (j, i) in edges:
    #             # already added
    #             return
    #         edges.add( (i, j) )
    #         edge_points.append(coords[ [i, j] ])

    #     coords = np.array([point.coords[0] for point in points])

    #     tri = Delaunay(coords)
    #     edges = set()
    #     edge_points = []
    #     # loop over triangles:
    #     # ia, ib, ic = indices of corner points of the triangle
    #     for ia, ib, ic in tri.vertices:
    #         pa = coords[ia]
    #         pb = coords[ib]
    #         pc = coords[ic]

    #         # Lengths of sides of triangle
    #         a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
    #         b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
    #         c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

    #         # Semiperimeter of triangle
    #         s = (a + b + c)/2.0

    #         # Area of triangle by Heron's formula
    #         area = math.sqrt(s*(s-a)*(s-b)*(s-c))
    #         circum_r = a*b*c/(4.0*area)

    #         # Here's the radius filter.
    #         # print(circum_r)
    #         if circum_r < 1000.0/alpha:
    #             add_edge(edges, edge_points, coords, ia, ib)
    #             add_edge(edges, edge_points, coords, ib, ic)
    #             add_edge(edges, edge_points, coords, ic, ia)
    #     m = sp.geometry.MultiLineString(edge_points)
    #     triangles = list(polygonize(m))
    #     self.concave_hull = unary_union(triangles)
    #     self.edge_points = edge_points
    #     self.plot_polygon(self.concave_hull)
    #     return self.concave_hull, self.edge_points
        
    # def optimizealpha(self):
    #     for i in range(10,13):
    #         alpha = round((i+1)*.1,2)
    #         print(alpha)
    #         concave_hull, edge_points = self.genalpha(alpha)

    #         #print concave_hull
    #         lines = LineCollection(edge_points)
    #         pl.figure(figsize=(20,10))
    #         pl.title('Alpha={0} Delaunay triangulation'.format(alpha))
    #         pl.gca().add_collection(lines)
    #         delaunay_points = np.array([point.coords[0] for point in self.points])
    #         pl.plot(delaunay_points[:,0], delaunay_points[:,1], 'o', color='#f13824')

    #         _ = self.plot_polygon(concave_hull)
            
    # def drawverticies(self):
    #         alpha = 1.3
    #         lines = LineCollection(self.edge_points)
    #         pl.figure(figsize=self.fs)
    #         pl.title('Alpha={0} Delaunay triangulation'.format(alpha))
    #         pl.gca().add_collection(lines)
    #         delaunay_points = np.array([point.coords[0] for point in self.points])
    #         pl.plot(delaunay_points[:,0], delaunay_points[:,1], 'o', color='#f13824')
        
    def process(self):
        self.maptopoints()
        #self.plot_points()
        #self.simpleconvexhull()
        #self.plot_polygon(self.point_collection.envelope)
        #self.plot_polygon(self.convex_hull_polygon)
        #self.optimizealpha()
        self.concave_hull, self.edge_points = self.genalpha2(1.3)
        #self.drawverticies()
        
class plotfun():
    
    def __init__(self,pg_cmp,diff_poly,fs=(20,10)):
        self.df = df
        self.dbf = False #debug flag
        self.fs = fs
        self.pg_cmp = pg_cmp
        self.diff_poly = diff_poly
        
    def plot_polygon(self,data):
        fig = pl.figure(figsize=self.fs)
        ax = fig.add_subplot(111)
        margin = 1
        x_min, y_min, x_max, y_max = data.bounds
        ax.set_xlim([x_min-margin, x_max+margin])
        ax.set_ylim([y_min-margin, y_max+margin])
        patch = PolygonPatch(data, fc='#999999', ec='#000000', fill=True, zorder=-1)
        ax.add_patch(patch)
        fig = pl.plot(self.x,self.y,'o', color='#f16824')
        return fig
    
    def plot_points(self):
        fig = pl.figure(figsize=self.fs)
        fig = pl.plot(self.x,self.y,'o', color='#f16824')
        return fig
    
    def plot_results(self):
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
        return fig
        
           
#%%
# Main script to run
# fn = r"E:\Scripting\dxf\RPL_P682_BSR.dxf"
fn = r"E:\Scripting\dxf\M86710-001_BSR_Cleaned Up.dxf"
# fn = r"E:\Scripting\dxf\input_example.dxf"
derp = loaddxf(fn)
derp.process()
df = derp.dfs
herp = generatemasks(derp.pg_cmp)
herp.process()
meow = plotfun(derp.pg_cmp,herp.diff_mask)
meow.plot_results()

# Scratch pages for testing
# # %%
from shapelysmooth import chaikin_smooth
#%%
cmp_mask[0]
smoothed_geometry = chaikin_smooth(cmp_mask[0], 5, keep_ends=False)
# %%
smoothed_geometry
# # %%

# %%
for cmp in cmp_mask:
    print(cmp.area)
# %%
