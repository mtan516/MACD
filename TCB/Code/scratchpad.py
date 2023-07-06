# Not used in final version but may be useful in the future# class generatepointmask():
    
    
    # def __init__(self,df,fs=(20,10)):
    #     self.df = df
    #     self.dbf = False #debug flag
    #     self.fs = fs
        
    # def maptopoints(self):
    #     self.points = gpd.GeoDataFrame(self.df, geometry=gpd.points_from_xy(self.df.X, self.df.Y)).reset_index(drop=True)
    #     self.points = self.points.geometry
    #     if self.dbf == True:
    #         print(self.points)
    #     self.x = [p.coords.xy[0] for p in self.points]
    #     self.y = [p.coords.xy[1] for p in self.points]
    #     self.point_collection = sp.geometry.MultiPoint(list(self.points))
    #     self.point_collection.envelope
    
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
        
    # def process(self):
    #     self.maptopoints()
    #     #self.plot_points()
    #     #self.simpleconvexhull()
    #     #self.plot_polygon(self.point_collection.envelope)
    #     #self.plot_polygon(self.convex_hull_polygon)
    #     #self.optimizealpha()
    #     self.concave_hull, self.edge_points = self.genalpha2(1.3)
    #     #self.drawverticies()# Scratch pages for testing
# # %%
from shapelysmooth import chaikin_smooth
#%%
cmp_mask[0]
smoothed_geometry = chaikin_smooth(meow, 5, keep_ends=False)
# %%
smoothed_geometry
# # %%