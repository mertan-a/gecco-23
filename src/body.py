import numpy as np
import _pickle as pickle

from evogym import get_full_connectivity, is_connected

triped = np.array([ 
          [3,3,3,3,3],
          [3,3,3,3,3],
          [3,0,3,0,3],
          [3,0,3,0,3],
          [3,0,3,0,3]
        ])

biped = np.array([ 
          [0,0,0,0,0],
          [3,3,3,3,3],
          [3,3,3,3,3],
          [3,3,0,3,3],
          [3,3,0,3,3]
        ])

block = np.array([
          [3,3,3,3,3],
          [3,3,3,3,3],
          [3,3,3,3,3],
          [3,3,3,3,3],
          [3,3,3,3,3]
        ])

worm = np.array([
          [0,0,0,0,0],
          [0,0,0,0,0],
          [0,0,0,0,0],
          [3,3,3,3,3],
          [3,3,3,3,3]
        ])

deneme = np.array([
            [3,0,3,3,0],
            [3,3,0,3,0],
            [0,3,3,3,0],
            [3,0,3,0,3],
            [3,3,3,3,3]
        ])

class BODY(object):
    def __init__(self, type):
        self.type = type
        self.nr_active_voxels = 0

    def mutate(self):
        raise NotImplementedError

    def to_phenotype(self):
        raise NotImplementedError

    def is_valid(self):
        raise NotImplementedError

    def count_active_voxels(self, structure):
        hs = np.sum( structure == 3 )
        vs = np.sum( structure == 4 )
        return hs + vs

class FIXED_BODY(BODY):
    def __init__(self, fixed_body=None, fixed_body_path=None):
        BODY.__init__(self, "fixed")
        self.bodies = []
        if fixed_body is not None:
            for body in fixed_body:
                if body == "biped":
                    structure = biped
                    name = "biped"
                elif body == "worm":
                    structure = worm
                    name = "worm"
                elif body == "block":
                    structure = block
                    name = "block"
                elif body == "deneme":
                    structure = deneme
                    name = "deneme"
                elif body == "triped":
                    structure = triped
                    name = "triped"
                else:
                    raise TypeError("Invalid fixed body")
                connections = get_full_connectivity(structure)
                self.bodies.append( {"structure": structure, "connections": connections, "name": name, "nr_active_voxels": self.count_active_voxels(structure)} )
                self.nr_active_voxels += self.count_active_voxels(structure)
        elif fixed_body_path is not None:
            self.bodies = self.extract_body(fixed_body_path)
            for body in self.bodies:
                self.nr_active_voxels += self.count_active_voxels(body["structure"])

    def mutate(self):
        raise TypeError("Cannot mutate fixed body")

    def is_valid(self):
        return True

    def extract_body(self, path_to_pkl):
        # path should be a .pkl file
        with open(path_to_pkl, 'rb') as f:
            ind = pickle.load(f)[0]
        return ind.body.bodies


class BASIC_BODY(BODY):
    def __init__(self, args):
        BODY.__init__(self, "evolvable")
        self.args = args
        self.structure = np.zeros((self.args.bounding_box[0], self.args.bounding_box[1]))
        self.mutate()

    def mutate(self):
        initial_structure = self.structure.copy()
        while np.sum(self.structure != 0) < self.structure.size*0.2 or np.all(self.structure == initial_structure) or np.sum(self.structure == 3) + np.sum(self.structure == 4) < 2:
            for i in range(self.structure.shape[0]):
                for j in range(self.structure.shape[1]):
                    if np.random.random() < 0.1:
                        copy_structure = self.structure.copy()
                        if copy_structure[i,j] == 0:
                            copy_structure[i,j] = np.random.choice([1,2,3,4])
                        elif copy_structure[i][j] == 1:
                            copy_structure[i][j] = np.random.choice([0,2,3,4])
                        elif copy_structure[i][j] == 2:
                            copy_structure[i][j] = np.random.choice([0,1,3,4])
                        elif copy_structure[i][j] == 3:
                            copy_structure[i][j] = np.random.choice([0,1,2,4])
                        else:
                            copy_structure[i][j] = np.random.choice([0,1,2,3])
                        if is_connected(copy_structure):
                            self.structure = copy_structure.copy()
        assert np.sum(self.structure) != 0, 'there is no body'
        # move the body to the bottom left
        while np.sum( self.structure[-1,:] ) == 0:
            self.structure = np.roll(self.structure, 1, axis=0)
        while np.sum( self.structure[:,0] ) == 0:
            self.structure = np.roll(self.structure, -1, axis=1)
        self.bodies = [ {"structure": self.structure, "connections": get_full_connectivity(self.structure), "name": "basic", "nr_active_voxels": self.count_active_voxels(self.structure)} ]
        self.nr_active_voxels = self.count_active_voxels(self.structure)

    def random_body(self):
        if self.args.single_material == True:
            return np.random.randint(0, 1, (self.args.bounding_box[0], self.args.bounding_box[1]))*3
        else:
            return np.random.randint(0, 5, (self.args.bounding_box[0], self.args.bounding_box[1]))

    def is_valid(self):
        return is_connected(self.bodies[0]["structure"])


if __name__ == '__main__':
    # Test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bounding_box', type=int, nargs=2, default=[5,5])
    args = parser.parse_args()
    test_body = BASIC_BODY(args)
    for i in range(10):
        test_body.mutate()
        print(test_body.bodies[0]["structure"])
        print(test_body.is_valid())

