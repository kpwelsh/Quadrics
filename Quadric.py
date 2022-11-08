import numpy as np

class Quadric:
    def __init__(self, Q):
        self.Q = Q
    
    def raycast(self, o, n):
        o = np.concatenate((o, [1]), axis=0)
        n = np.concatenate((n, [0]), axis=0)
        qn = self.Q @ n
        qo = self.Q @ o
        a = n.dot(qn)
        b = o.dot(qn) + n.dot(qo)
        c = o.dot(qo)

        desc = b**2 - 4 * a * c
        if abs(a) < 1e-8:
            return [c / (-b)]
        if desc < 0:
            return []
        sa = np.sign(a)
        return [
            (-b - sa * np.sqrt(desc)) / (2 * a),
            (-b + sa * np.sqrt(desc)) / (2 * a),
        ]

class Paraboloid(Quadric):
    '''
        Locally defined by z = a x^2 + b y^2.
        Globally defined with a transform
    '''
    def __init__(self, a, b):
        super().__init__(
            np.array((
                (a, 0, 0, 0),
                (0, b, 0, 0),
                (0, 0, 0, -1),
                (0, 0, 0, 0),
            ))
        )

class Sphere(Quadric):
    '''
        Locally defined by x^2 + y^2 + z^2 = r^2
        Globally defined with a transform
    '''
    def __init__(self, r):
        super().__init__(
            np.array((
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1., 0),
                (0, 0, 0, -r**2),
            ))
        )

class Ellipse(Quadric):
    '''
        Locally defined by a x^2 + b y^2 + c z^2 = r^2
        Globally defined with a transform
    '''
    def __init__(self, a, b, c, r):
        super().__init__(
            np.array((
                (a, 0, 0, 0),
                (0, b, 0, 0),
                (0, 0, c, 0),
                (0, 0, 0, r**2),
            ))
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ax = plt.subplot(projection='3d')

    theta = np.linspace(0, np.pi, 100, endpoint=True)
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=True)

    THETA, PHI = np.meshgrid(theta, phi)
    ns = []
    for t, p in np.column_stack((THETA.flatten(), PHI.flatten())):
        ns.append(np.array((
            np.sin(t) * np.cos(p),
            np.sin(t) * np.sin(p),
            np.cos(t)
        )))
    
    o = np.array((0, 0, 10))

    p = Paraboloid(1, 1)
    sphere = Sphere(2)
    points = []
    for n in ns:
        for t in p.raycast(o, n):
            if abs(t) < 100:
                points.append(o + t * n)
    
    points = np.array(points)
    ax.scatter(*points.T)
    plt.show()
    