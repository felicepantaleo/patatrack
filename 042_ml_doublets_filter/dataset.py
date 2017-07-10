import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


datalabs = ["run", "evt", "detSeqIn", "detSeqOut", "inZ", "inX", "inY", "outZ",
            "outX", "outY", "detCounterIn", "detCounterOut", "isBarrelIn", "isBarrelOut",
            "layerIn", "ladderIn", "moduleIn", "sideIn", "diskIn", "panelIn", "bladeIn",
            "layerOut", "ladderOut", "moduleOut", "sideOut", "diskOut", "panelOut", "bladeOut",
            "inId", "outId", "isBigIn", "isEdgIn", "isBadIn", "isBigOut", "isEdgOut", "isBadOut", "isFlippedIn", "isFlippedOut",
            "inPix1", "inPix2", "inPix3", "inPix4", "inPix5", "inPix6", "inPix7", "inPix8",
            "inPix9", "inPix10", "inPix11", "inPix12", "inPix13", "inPix14", "inPix15", "inPix16",
            "inPix17", "inPix18", "inPix19", "inPix20", "inPix21", "inPix22", "inPix23", "inPix24",
            "inPix25", "inPix26", "inPix27", "inPix28", "inPix29", "inPix30", "inPix31", "inPix32",
            "inPix33", "inPix34", "inPix35", "inPix36", "inPix37", "inPix38", "inPix39", "inPix40",
            "inPix41", "inPix42", "inPix43", "inPix44", "inPix45", "inPix46", "inPix47", "inPix48",
            "inPix49", "inPix50", "inPix51", "inPix52", "inPix53", "inPix54", "inPix55", "inPix56",
            "inPix57", "inPix58", "inPix59", "inPix60", "inPix61", "inPix62", "inPix63", "inPix64",
            "outPix1", "outPix2", "outPix3", "outPix4", "outPix5", "outPix6", "outPix7", "outPix8",
            "outPix9", "outPix10", "outPix11", "outPix12", "outPix13", "outPix14", "outPix15", "outPix16",
            "outPix17", "outPix18", "outPix19", "outPix20", "outPix21", "outPix22", "outPix23", "outPix24",
            "outPix25", "outPix26", "outPix27", "outPix28", "outPix29", "outPix30", "outPix31", "outPix32",
            "outPix33", "outPix34", "outPix35", "outPix36", "outPix37", "outPix38", "outPix39", "outPix40",
            "outPix41", "outPix42", "outPix43", "outPix44", "outPix45", "outPix46", "outPix47", "outPix48",
            "outPix49", "outPix50", "outPix51", "outPix52", "outPix53", "outPix54", "outPix55", "outPix56",
            "outPix57", "outPix58", "outPix59", "outPix60", "outPix61", "outPix62", "outPix63", "outPix64",
            "dummyFlag", "idTrack", "px", "py", "pz", "pt", "mT", "eT", "mSqr", "rapidity", "etaTrack", "phi",
            "pdgId", "charge", "noTrackerHits", "noTrackerLayers", "dZ", "dXY", "Xvertex",
            "Yvertex", "Zvertex", "bunCross", "isCosmic", "chargeMatch", "sigMatch"
            ]

inhitlabs = ["inPix1", "inPix2", "inPix3", "inPix4", "inPix5", "inPix6", "inPix7", "inPix8",
             "inPix9", "inPix10", "inPix11", "inPix12", "inPix13", "inPix14", "inPix15", "inPix16",
             "inPix17", "inPix18", "inPix19", "inPix20", "inPix21", "inPix22", "inPix23", "inPix24",
             "inPix25", "inPix26", "inPix27", "inPix28", "inPix29", "inPix30", "inPix31", "inPix32",
             "inPix33", "inPix34", "inPix35", "inPix36", "inPix37", "inPix38", "inPix39", "inPix40",
             "inPix41", "inPix42", "inPix43", "inPix44", "inPix45", "inPix46", "inPix47", "inPix48",
             "inPix49", "inPix50", "inPix51", "inPix52", "inPix53", "inPix54", "inPix55", "inPix56",
             "inPix57", "inPix58", "inPix59", "inPix60", "inPix61", "inPix62", "inPix63", "inPix64"
             ]

outhitlabs = ["outPix1", "outPix2", "outPix3", "outPix4", "outPix5", "outPix6", "outPix7", "outPix8",
              "outPix9", "outPix10", "outPix11", "outPix12", "outPix13", "outPix14", "outPix15", "outPix16",
              "outPix17", "outPix18", "outPix19", "outPix20", "outPix21", "outPix22", "outPix23", "outPix24",
              "outPix25", "outPix26", "outPix27", "outPix28", "outPix29", "outPix30", "outPix31", "outPix32",
              "outPix33", "outPix34", "outPix35", "outPix36", "outPix37", "outPix38", "outPix39", "outPix40",
              "outPix41", "outPix42", "outPix43", "outPix44", "outPix45", "outPix46", "outPix47", "outPix48",
              "outPix49", "outPix50", "outPix51", "outPix52", "outPix53", "outPix54", "outPix55", "outPix56",
              "outPix57", "outPix58", "outPix59", "outPix60", "outPix61", "outPix62", "outPix63", "outPix64"
              ]

featurelabs = ["detSeqIn", "detSeqOut", "inZ", "inX", "inY",
            "outZ", "outX", "outY", "detCounterIn", "detCounterOut", "isBarrelIn", "isBarrelOut",
            "layerIn", "ladderIn", "moduleIn", "sideIn", "diskIn", "panelIn", "bladeIn",
            "layerOut", "ladderOut", "moduleOut", "sideOut", "diskOut", "panelOut", "bladeOut",
            "inId", "outId", "isBigIn", "isEdgIn", "isBadIn", "isBigOut", "isEdgOut", "isBadOut", "isFlippedIn", "isFlippedOut",
            ]
targetlabes = ["dummyFlag", "idTrack", "px", "py", "pz", "pt", "mT", "eT", "mSqr", "rapidity", "etaTrack", "phi",
            "pdgId", "charge", "noTrackerHits", "noTrackerLayers", "dZ", "dXY", "Xvertex",
            "Yvertex", "Zvertex", "bunCross", "isCosmic", "chargeMatch", "sigMatch"
            ]

class Dataset:
    """ Load the dataset from txt files. """
    def __init__(self, fname, delimit='\t'):
        with open(fname, 'rb') as f:
            data = np.genfromtxt(f, delimiter=delimit, dtype=np.float32)
            self.data = pd.DataFrame(data, columns=datalabs)

    def get_hit_rectangle(self):
        """ Returns hits clusters data for Adriano's model. """
        a = self.data[inhitlabs + outhitlabs].as_matrix()
        return a

    def get_hit_shapes(self, normalize=True):
        """ Return hit shape features
        Args:
        -----
            normalize : (bool)
                normalize the data matrix with zero mean and unitary variance.
        """
        a_in = self.data[inhitlabs].as_matrix()
        a_out = self.data[outhitlabs].as_matrix()
        
        # Normalize data
        if normalize:
            self.scaler_in = StandardScaler()
            a_in = self.scaler_in.fit_transform(a_in)
            self.scaler_out = StandardScaler()
            a_out = self.scaler_out.fit_transform(a_out)

        data = np.array([a_in, a_out]) # (2, batch_size, hit_size)
        data = data.reshape((2, -1, 8, 8))
        return np.transpose(data, (1, 2, 3, 0))

    def get_info_features(self):
        """ Returns info features as numpy array. """
        return self.data[featurelabs]

    def get_labels(self):
        return self.data["pdgId"].as_matrix() != 0.0


if __name__ == '__main__':
    d = Dataset('data/1_1_5_dataset.txt')
    batch_size = d.data.as_matrix().shape[0]

    x = d.get_hit_shapes()
    assert x.shape == (batch_size, 8, 8, 2)
    np.testing.assert_allclose(x[:, :,:,0], d.data[inhitlabs].as_matrix().reshape((-1, 8, 8)))

    print("All test successfully completed.")

