import itertools
import subprocess
import pickle
import numpy as np
from sac.util import Util
import os
import feat
from sklearn.svm import SVC

DATASETS = os.path.abspath("./training_dataset")
FEATURE_PLAN = os.path.abspath("./")

features = Util.read_feature_names_from_file(os.path.join(FEATURE_PLAN, "featureplan"))

features1 = ["zcr", "flux", "spectral_rollof", "energy_stats"]
features2 = ["mfcc_stats"]
features3 = ["spectral_flatness_per_band"]
features4 = features1 + features2 + features3

n = len(os.listdir(DATASETS))

print " - Computing features..."

for i in xrange(n):
    print "   - Training part %d/%d" % ((i + 1), n)
    cmd = ["yaafe", "-c", os.path.join(FEATURE_PLAN, "featureplan"), "-r", "22050", DATASETS + "/audio_{0}/audio_{0}_combined.wav".format(i)]
    subprocess.check_output(cmd)

print " - Loading features..."

data = feat.get_features(features4, DATASETS, n, pca=False)

model = SVC(probability=True)
X = np.vstack((data["x_" + str(i)] for i in xrange(n)))
Y = list(itertools.chain.from_iterable([data["y_" + str(i)] for i in xrange(n)]))

print " - Training model..."

model.fit(X, Y)

print "DONE"

with open("model/model_.pickle", 'w') as f:
    pickle.dump(model, f)
