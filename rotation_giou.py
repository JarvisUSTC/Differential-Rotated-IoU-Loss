from math import pi, cos, sin
import torch
from torch.autograd import Variable
import shapely
from shapely.geometry import Polygon, MultiPoint
import numpy as np

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )


def rectangle_vertices(rotated_rectangle):
    cx, cy, w, h, r = rotated_rectangle
    angle = pi*r/180
    dx = w/2
    dy = h/2
    dxcos = dx*cos(angle)
    dxsin = dx*sin(angle)
    dycos = dy*cos(angle)
    dysin = dy*sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos - -dysin,  dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos -  dysin,  dxsin +  dycos),
        Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
    )

def intersection_area(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices
    if len(r1) == 5 and len(r2) == 5:
        rect1 = rectangle_vertices(r1)
        rect2 = rectangle_vertices(r2)
    elif len(r1) == 8 and len(r2) == 8:
        rect1 = [Vector(vertex[0],vertex[1]) for vertex in r1.view(-1,2)]
        rect2 = [Vector(vertex[0],vertex[1]) for vertex in r2.view(-1,2)]

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]

def convex_hull_area(points):
    convex_hull_vertex = convex_hull(points)
    return 0.5 * sum(px*qy - py*qx for (px,py), (qx,qy) in
                     zip(convex_hull_vertex, convex_hull_vertex[1:] + convex_hull_vertex[:1]))

# Example: convex hull of a 10-by-10 grid.
assert convex_hull([(i//10, i%10) for i in range(100)]) == [(0, 0), (9, 0), (9, 9), (0, 9)]

def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """

 

    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

 

    point = [
                [int(points[0]) , int(points[1])],
                [int(points[2]) , int(points[3])],
                [int(points[4]) , int(points[5])],
                [int(points[6]) , int(points[7])]
            ]
    edge = [
                ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
                ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
                ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
                ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
    ]

 

    summatory = edge[0] + edge[1] + edge[2] + edge[3]
    if summatory>0:
        return False
    else:
        return True

def clockwise_sort(points):
    from functools import reduce
    import operator
    import math
    coords = points.reshape((-1, 2))
    center = torch.mean(coords, dim=0).reshape((1, 2))
    sorted_coords = sorted(coords, key=lambda coord: -(-135 - math.degrees(math.atan2((coord - center)[0,1], (coord - center)[0,0]))) % 360)
    sorted_coords = torch.cat(sorted_coords).reshape(-1)
    # assert validate_clockwise_points(sorted_coords)
    return sorted_coords

def GIoU_Rotated_Rectangle(Box1,Box2):
    """Compute the differentiable giou between two arbitrary rotated rectangles.
    GIoU(b1,b2) = IoU(b1,b2) - area(C\(b1 union b2))/area(C) = area(b1 intersection b2)/area(b1 union b2) - area(C\(b1 union b2))/area(C)
    area(b1 union b2) = area(b1) + area(b2) - area(b1 intersection b2)
    area(C) = area(convex_hull)

    Input: Box1, Box2: (cx,cy,w,h,angle), torch.tensor, requires_grad = true
    Output: GIoU: torch.tensor
    """
    if len(Box1) == 5:
        area_box1 = Box1[2]*Box1[3] # for rectangle
        area_box2 = Box2[2]*Box2[3]
        box1_vertex = [(vertex.x,vertex.y)for vertex in rectangle_vertices(Box1)]
        box2_vertex = [(vertex.x,vertex.y)for vertex in rectangle_vertices(Box2)]
    elif len(Box1) == 8:
        # if not validate_clockwise_points(Box1):
        Box1 = clockwise_sort(Box1)
        # if not validate_clockwise_points(Box2):
        Box2 = clockwise_sort(Box2)
        box1_vertex = [(vertex[0],vertex[1]) for vertex in Box1.view(-1,2)]
        box2_vertex = [(vertex[0],vertex[1]) for vertex in Box2.view(-1,2)]
        
        area_box1 = 0.5 * sum(px*qy - py*qx for (px,py), (qx,qy) in
                     zip(box1_vertex, box1_vertex[1:] + box1_vertex[:1]))
        area_box2 = 0.5 * sum(px*qy - py*qx for (px,py), (qx,qy) in
                     zip(box2_vertex, box2_vertex[1:] + box2_vertex[:1]))
    area_box1_intersection_box2 = intersection_area(Box1, Box2)
    all_vertex = box1_vertex + box2_vertex
    C = convex_hull_area(all_vertex)
    area_box1_union_box2 = area_box1 + area_box2 - area_box1_intersection_box2
    GIoU = area_box1_intersection_box2/area_box1_union_box2 - (C-area_box1_union_box2)/C
    return GIoU

# shapely package implementation for giou
def polygon_generalized_box_iou(boxes1, boxes2):
    """
    polygon giou
    Args:
        boxes1: list((x1, y1, x2, y2, x3, y3, x4, y4))
        boxes2: list((x1, y1, x2, y2, x3, y3, x4, y4))

    Returns: giou_matrix

    """
    giou_matrix = torch.zeros((len(boxes1),len(boxes2)),dtype=torch.float32).to(boxes1.device)
    for i,box1 in enumerate(boxes1):
        for j,box2 in enumerate(boxes2):
            Polygon1 = Polygon(box1.view(4,-1)).convex_hull
            Polygon2 = Polygon(box2.view(4,-1)).convex_hull
            intersection_area = float(Polygon1.intersection(Polygon2).area)
            union_area = float(Polygon1.union(Polygon2).area)
            convex_hull_2 = Polygon(torch.cat((box1.view(4,-1),box2.view(4,-1)),0)).convex_hull
            iou = intersection_area/union_area
            giou = iou - (float(convex_hull_2.area)-union_area) / float(convex_hull_2.area)
            giou_matrix[i,j] = giou

    return giou_matrix


if __name__ == '__main__':
    r2 = (10, 10, 20, 20, -30)
    r1 = (10, 10, 10, 10, 30)
    Box1 = Variable(torch.tensor([10, 10, 20, 20, -30],dtype=float),requires_grad=True)
    Box2 = Variable(torch.tensor([10, 100, 10, 10, 30],dtype=float),requires_grad=True)
    boxes1 = []
    boxes2 = []
    for vertex in rectangle_vertices(Box1):
        boxes1 += [vertex.x,vertex.y]
    for vertex in rectangle_vertices(Box2):
        boxes2 += [vertex.x,vertex.y]
    print("Two rectangle using (cx,cy,w,h,angle) representation:",Box1,Box2)
    print("using shapely package:",polygon_generalized_box_iou(torch.tensor([boxes1]),torch.tensor([boxes2])))
    # print(intersection_area(r1, r2))
    print("using my giou implementation",GIoU_Rotated_Rectangle(Box1,Box2))
    GIoU_Rotated_Rectangle(Box1,Box2).backward()
    print("the grad of two rectangle after backward:",Box1.grad,Box2.grad)
    Box3 = Variable(torch.tensor([0,0,10,5,5,10,0,8],dtype=float),requires_grad=True)
    Box4 = Variable(torch.tensor([0,0,15,5,5,10,0,8],dtype=float),requires_grad=True)
    print("Two polygon using (x1,y1,x2,y2,x3,y3,x4,y4) representation:",Box3,Box4)
    print("using shapely package:",polygon_generalized_box_iou(Box3.unsqueeze(0),Box4.unsqueeze(0)))
    print("using my giou implementation for polygon",GIoU_Rotated_Rectangle(Box3, Box4))
    GIoU_Rotated_Rectangle(Box3, Box4).backward()
    print("the grad of two polygon after backward:",Box3.grad,Box4.grad)
    Box5 = Variable(torch.tensor([0,0,0,8,5,10,10,5],dtype=float),requires_grad=True)
    Box6 = Variable(torch.tensor([0,0,0,8,5,10,15,5],dtype=float),requires_grad=True)
    print("Two polygon using (x1,y1,x2,y2,x3,y3,x4,y4) representation:",Box5,Box6)
    print("using shapely package:",polygon_generalized_box_iou(Box5.unsqueeze(0),Box6.unsqueeze(0)))
    print("using my giou implementation for polygon",GIoU_Rotated_Rectangle(Box5, Box6))
    GIoU_Rotated_Rectangle(Box5, Box6).backward()
    print("the grad of two polygon after backward:",Box5.grad,Box6.grad)