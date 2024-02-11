import numpy as np
import math
import cmath


class View:

    def __init__(self, init_pos, center, t_1):
        self.init_pos = init_pos
        self.t = None
        self.get_next_camera_pos(center, t_1)

    def get_next_camera_pos(self, object_center_world, t_1):
        Tworld2center = np.eye(4)
        Tworld2center[:3, 3] = object_center_world

        object = np.array(object_center_world)
        view = np.array(self.init_pos)

        Z = object - view
        Z /= np.linalg.norm(Z)

        X = np.cross(Z, view)
        X /= np.linalg.norm(X)

        Y = np.cross(Z, X)
        Y /= np.linalg.norm(Y)

        T = np.eye(4)
        T[:3, 3] = view

        R = np.zeros((4, 4))
        R[:3, :3] = np.column_stack((X, Y, Z))
        R[3, 3] = 1

        h = np.dot(T, R)
        self.t = h

        phi_min = self.minimize_Rzangle(h[:3, 0], h[:3, 1], h[:3, 2], t_1[0, 0], t_1[1, 0], t_1[2, 0])
        self.t = np.dot(self.t, np.array([[math.cos(phi_min + math.pi / 2), -math.sin(phi_min + math.pi / 2), 0, 0],
                                          [math.sin(phi_min + math.pi / 2), math.cos(phi_min + math.pi / 2), 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]]))

        angle = math.acos(self.t[1, 0])

        if angle > math.pi / 2:
            self.t = np.dot(self.t, np.array([[-1, 0, 0, 0],
                                              [0, -1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]]))

    def minimize_Rzangle(self, ra11, ra21, ra31, ra12, ra22, ra32, rb11, rb21, rb31):
        real_num = -rb21 * ra11 * ra12 * ra22 - rb31 * ra11 * ra12 * ra32 + rb11 * ra11 * ra22 ** 2 + \
                   rb11 * ra11 * ra32 ** 2 + rb21 * ra12 ** 2 * ra21 + rb31 * ra12 ** 2 * ra31 - \
                   rb11 * ra12 * ra21 * ra22 - rb11 * ra12 * ra31 * ra32 - rb31 * ra21 * ra22 * ra32 + \
                   rb21 * ra21 * ra32 ** 2 + rb31 * ra22 ** 2 * ra31 - rb21 * ra22 * ra31 * ra32

        imag_num = rb21 * ra11 ** 2 * ra22 + rb31 * ra11 ** 2 * ra32 - rb21 * ra11 * ra12 * ra21 - \
                   rb31 * ra11 * ra12 * ra31 - rb11 * ra11 * ra21 * ra22 - rb11 * ra11 * ra31 * ra32 + \
                   rb11 * ra12 * ra21 ** 2 + rb11 * ra12 * ra31 ** 2 + rb31 * ra21 ** 2 * ra32 - \
                   rb31 * ra21 * ra22 * ra31 - rb21 * ra21 * ra31 * ra32 + rb21 * ra22 * ra31 ** 2

        imag_den = -rb21 * ra11 ** 2 * ra22 - rb31 * ra11 ** 2 * ra32 + rb21 * ra11 * ra12 * ra21 + \
                   rb31 * ra11 * ra12 * ra31 + rb11 * ra11 * ra21 * ra22 + rb11 * ra11 * ra31 * ra32 - \
                   rb11 * ra12 * ra21 ** 2 - rb11 * ra12 * ra31 ** 2 - rb31 * ra21 ** 2 * ra32 + \
                   rb31 * ra21 * ra22 * ra31 + rb21 * ra21 * ra31 * ra32 - rb21 * ra22 * ra31 ** 2

        real_den = -rb21 * ra11 * ra12 * ra22 - rb31 * ra11 * ra12 * ra32 + rb11 * ra11 * ra22 ** 2 + \
                   rb11 * ra11 * ra32 ** 2 + rb21 * ra12 ** 2 * ra21 + rb31 * ra12 ** 2 * ra31 - \
                   rb11 * ra12 * ra21 * ra22 - rb11 * ra12 * ra31 * ra32 - rb31 * ra21 * ra22 * ra32 + \
                   rb21 * ra21 * ra32 ** 2 + rb31 * ra22 ** 2 * ra31 - rb21 * ra22 * ra31 * ra32

        numerador = complex(real_num, imag_num)
        denominador = complex(real_den, imag_den)

        sustraendo_1 = cmath.log(numerador / denominador)
        sustraendo_2 = complex(0.0, 1.0)

        producto = sustraendo_1 * sustraendo_2
        sustraendo = complex(producto.real / 2, producto.imag / 2)

        min_angle = cmath.pi - sustraendo
        return min_angle.real


class View_Space:

    def __init__(self, center, r):
        self.object_center_world = center
        self.radio = r
        self.num_views = 32
        self.pt_sphere = []
        self.views = []

        self.get_view_space()

    def get_view_space(self):
        t_1 = np.eye(4)
        for i in range(self.num_views):
            view = View([self.pt_sphere[i][0], self.pt_sphere[i][1], self.pt_sphere[i][2]], self.object_center_world, t_1)
            self.views.append(view)

    def update(self, center, r):
        self.pt_sphere.clear()
        self.views.clear()

        self.object_center_world = center
        self.radio = r
        self.num_views = 32

        # Load pt_sphere data from file
        with open("tot_sort.txt", "r") as fin_sphere:
            for i in range(self.num_views):
                pt = [float(val) for val in fin_sphere.readline().split()]
                px, py, pz = pt
                self.pt_sphere.append([
                    self.radio * (px / math.sqrt(px ** 2 + py ** 2 + pz ** 2)) + self.object_center_world[0],
                    self.radio * (py / math.sqrt(px ** 2 + py ** 2 + pz ** 2)) + self.object_center_world[1],
                    self.radio * (pz / math.sqrt(px ** 2 + py ** 2 + pz ** 2)) + self.object_center_world[2]
                ])

        self.get_view_space()

    def views2tStamped(self):
        tfStamped_vector = []

        for i, view in enumerate(self.views):
            v_rpy = view.getRPY()
            v_xyz = view.t.translation()

            tS = {
                "header": {
                    "frame_id": "panda_link0",
                    "stamp": {
                        "secs": 0,
                        "nsecs": 0
                    }
                },
                "child_frame_id": "tf_d" + str(i),
                "transform": {
                    "translation": {
                        "x": v_xyz[0],
                        "y": v_xyz[1],
                        "z": v_xyz[2]
                    },
                    "rotation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0
                    }
                }
            }

            q = tf2.Quaternion()
            q.setRPY(v_rpy[0], v_rpy[1], v_rpy[2])
            tS["transform"]["rotation"]["x"] = q.x()
            tS["transform"]["rotation"]["y"] = q.y()
            tS["transform"]["rotation"]["z"] = q.z()
            tS["transform"]["rotation"]["w"] = q.w()

            tfStamped_vector.append(tS)

        return tfStamped_vector


if __name__ == "__main__":
    # Sample object center and radius
    object_center = np.array([0.0, 0.0, 1.0])
    radius = 0.5

    # Create a View_Space object
    view_space = View_Space(object_center, radius)

    # Get the list of views as TransformStamped objects
    tfStamped_vector = view_space.views2tStamped()

    # Print the TransformStamped information for each view
    for i, tfStamped in enumerate(tfStamped_vector):
        print(f"View {i}:")
        print(f"Translation (x, y, z): ({tfStamped['transform']['translation']['x']}, "
              f"{tfStamped['transform']['translation']['y']}, {tfStamped['transform']['translation']['z']})")
        print(f"Rotation (x, y, z, w): ({tfStamped['transform']['rotation']['x']}, "
              f"{tfStamped['transform']['rotation']['y']}, {tfStamped['transform']['rotation']['z']}, "
              f"{tfStamped['transform']['rotation']['w']})")
        print("\n")

