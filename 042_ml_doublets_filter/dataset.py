import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical


target_lab = "pdgId"
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
    def __init__(self, fname, delimit='\t', max_rows=None):
        with open(fname, 'rb') as f:
            data = np.load(f)
            # Compressed .npz files return a dictionary
            if type(data) != np.ndarray:
                data = data['arr_0']
            if max_rows:
                data = data[:max_rows, :]
            self.data = pd.DataFrame(data, columns=datalabs)

    def phi_correction(self, hits_in, hits_out):
        #phi correction
        cosPhiIns = np.cos(np.arctan(np.multiply(self.data["inY"], 1.0/self.data["inX"])))
        cosPhiOuts = np.cos(np.arctan(np.multiply(self.data["outY"], 1.0/self.data["outX"])))
        sinPhiIns = np.sin(np.arctan(np.multiply(self.data["inY"], 1.0/self.data["inX"])))
        sinPhiOuts = np.sin(np.arctan(np.multiply(self.data["outY"], 1.0/self.data["outX"])))

        inPhiModC = np.multiply(hits_in,cosPhiIns[:,np.newaxis])
        outPhiModC = np.multiply(hits_out,cosPhiOuts[:,np.newaxis])

        inPhiC  = hits_in  + inPhiModC
        outPhiC = hits_out + outPhiModC
        return inPhiC, outPhiC

    def theta_correction(self, hits_in, hits_out):
        #theta correction
        cosThetaIns = np.cos(np.arctan(np.multiply(self.data["inY"],1.0/self.data["inZ"])))
        cosThetaOuts = np.cos(np.arctan(np.multiply(self.data["outY"],1.0/self.data["outZ"])))
        sinThetaIns = np.sin(np.arctan(np.multiply(self.data["inY"],1.0/self.data["inZ"])))
        sinThetaOuts = np.sin(np.arctan(np.multiply(self.data["outY"],1.0/self.data["outZ"])))

        inThetaModC = np.multiply(hits_in,cosThetaIns[:,np.newaxis])
        outThetaModC = np.multiply(hits_out,cosThetaOuts[:,np.newaxis])

        inThetaC  = hits_in  + inThetaModC
        outThetaC = hits_out + outThetaModC
        return inThetaC, outThetaC

    def get_hit_shapes(self, normalize=True, angular_correction=True):
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
            mean, std = (668.25684, 3919.5576) # mean, std precomputed for data NOPU 
            a_in = (a_in - mean) / std
            a_out = (a_out - mean) / std

        l = [a_in, a_out]        
        if angular_correction:
            phi_in, phi_out = self.phi_correction(a_in, a_out)
            theta_in, theta_out = self.theta_correction(a_in, a_out)
            data = l + [phi_in, phi_out, theta_in, theta_out]

        data = np.array(l) # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, 8, 8))
        return np.transpose(data, (1, 2, 3, 0))  # TODO: not optimal for CPU execution

    def filter(self, feature_name, value):
        """ filter data keeping only those samples where s[feature_name] = value """
        self.data = self.data[self.data[feature_name] == value]
        return self  # to allow method chaining

    def get_info_features(self):
        """ Returns info features as numpy array. """
        return self.data[featurelabs].as_matrix()

    def get_labels(self):
        return self.data[target_lab].as_matrix() != 0.0

    def get_data(self, normalize=True):
        X_hit = self.get_hit_shapes(normalize=normalize)
        X_info = self.get_info_features()
        y = to_categorical(self.get_labels())
        return X_hit, X_info, y

    def save(self, fname):
        np.save(fname, self.data.as_matrix())

    # TODO: pick doublets from same event.
    def balance_data(self, max_ratio=0.5, verbose=True):
        """ Balance the data. """ 
        data_neg = self.data[self.data[target_lab] == 0.0]
        data_pos = self.data[self.data[target_lab] != 0.0]

        n_pos = data_pos.shape[0]
        n_neg = data_neg.shape[0]
        if verbose:
            print("Number of negatives: " + str(n_neg))
            print("Number of positive: " + str(n_pos))
            print("ratio: " + str(n_neg / n_pos))
        
        data_neg = data_neg.sample(n_pos)
        balanced_data = pd.concat([data_neg, data_pos])
        balanced_data = balanced_data.sample(frac=1)  # Shuffle the dataset again
        self.data = balanced_data


if __name__ == '__main__':
    d = Dataset('data/debug.npy')
    batch_size = d.data.as_matrix().shape[0]

    x = d.get_data(normalize=False)[0]
    assert x.shape == (batch_size, 8, 8, 2)
    np.testing.assert_allclose(x[:, :,:,0], d.data[inhitlabs].as_matrix().reshape((-1, 8, 8)))

    print("All test successfully completed.")

