# encoding: utf-8
"""
Script to run RSA on a searchlight pattern.

13.01. 2016 run as 
%run RSA_searchlight.py -o /neuro/data/redness/Redness1_data/data_for_figures/rsa/rsa-aalto85_ -abscon -c /neuro/data/redness/semantic_features/PsychoLingVariables/ControlVariablesSet2_20161208.mat neuro/data/redness/semantic_features/AaltoNorms/aalto85/Aalto85_sorted20160204.mat

7-8.12. 2016 run as 
%run RSA_searchlight.py -o /neuro/data/redness/Redness1_data/data_for_figures/rsa/rsa-aalto85_ -abscon /neuro/data/redness/semantic_features/AaltoNorms/aalto85/Aalto85_sorted20160204.mat

8.12. 2016 run as 
%run RSA_searchlight.py -o /neuro/data/redness/Redness1_data/data_for_figures/rsa/rsa-ginter10 -abscon /neuro/data/redness/semantic_features/Ginter/Ginter-300-10+10.mat

15.12 2016 run as 
%run RSA_searchlight.py -o /neuro/data/redness/Redness1_data/data_for_figures/rsa/rsa-aalto85_2_ -abscon /neuro/data/redness/semantic_features/AaltoNorms/aalto85/Aalto85_sorted20160204.mat
%run RSA_searchlight.py -o /neuro/data/redness/Redness1_data/data_for_figures/rsa/s01-rsa-aalto85_ -i /neuro/data/redness/Redness1_data/source_localized/ico4/s01/PCA/dspm_morphed/s01-ico4-dSPMmorphed- /neuro/data/redness/semantic_features/AaltoNorms/aalto85/Aalto85_sorted20160204.mat

16.12 2016 run as 
%run RSA_searchlight.py -o /neuro/data/redness/Redness1_data/data_for_figures/rsa/rsa-aalto85_2_  /neuro/data/redness/semantic_features/AaltoNorms/aalto85/Aalto85_sorted20160204.mat

19.12 2016 run as
%run RSA_searchlight.py -ds -i /neuro/data/redness/Redness1_data/source_localized/ico4/s01/PCA/dspm_morphed/s01-ico4-dSPMmorphed- -o /neuro/data/redness/Redness1_data/data_for_figures/rsa/s01-rsa-aalto85_ /neuro/data/redness/semantic_features/AaltoNorms/aalto85/Aalto85_sorted20160204.mat

Authors
-------
Marijn van Vliet
Annika Hultén
"""

import numpy as np
import mne
import argparse
import os
import re
from scipy.io import loadmat
from scipy.spatial import distance
from scipy import stats
import progressbar
import subprocess
from sklearn import linear_model

print('Code version:'+ subprocess.getoutput('git rev-parse HEAD'))

# Default path where the grand average STC files for each word are stored
datapath = '/neuro/data/redness/Redness1_data/source_localized/ico4/grand_average/word_stc/'

# Default path for the source space
source_space_path = '/m/nbe/project/redness1/mri/fsaverage-5.1.0/bem/fsaverage-5.1.0-ico-4-src.fif'

# Deal with command line arguments
parser = argparse.ArgumentParser(description='Perform RSA analysis on Redness1 data.')
parser.add_argument('norms', type=str,
                    help='The file that contains the norm data; should be a .mat file.')
parser.add_argument('-i', '--input_folder', metavar='filename', type=str, default=datapath,
                    help='The folder that contains the subject specific stc files for each word. Defaults to %s' % datapath)
parser.add_argument('-o', '--output', metavar='filename', type=str, default='./rsa',
                    help='The file to write the results to, without the -(lh/rh).stc part. Defaults to ./rsa')
parser.add_argument('-a', '--abscon', action='store_true',
                    help='Whether to compute separate RSAs for abstract and concrete words.')
parser.add_argument('-ds', '--down_sample', action='store_true',
                    help='Whether to downsample with 20 ms window using data between 0-800 ms')
parser.add_argument('-j', '--jobs', type=int, default=2,
                    help='Number of jobs (=cores) to use during computation. Defaults to 2.')
parser.add_argument('-c', '--control', metavar='filename', type=str, default=None,
                    help='The file that contains the control variables; should end in .mat. Defaults to None, which disables adding control variables.')
parser.add_argument('-s', '--source', metavar='filename', type=str, default=source_space_path,
                    help='The file that contains the source space. Defaults to %s.' % source_space_path)
parser.add_argument('-b', '--break-after', metavar='N', type=int, default=None,
                    help='If set, break after N iterations')
parser.add_argument('-m', '--metric', type=str, default='correlation',
                    help='The distance metric to use when computing the DSM for the norms. Defaults to "correlation". Note that this does not chance the distance metric for the brain data.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Display a progress bar.')
args = parser.parse_args()

print 'Norms:', args.norms
print 'Input folder:', args.input_folder
print 'Output:', args.output
if args.abscon:
    print 'Absolute and concrete items analyzed separately'
if args.control is not None:
    print 'Control variables:', args.control

if args.verbose:
    from tqdm import tqdm

# Some parameters that we don't expose as command line arguments
spatial_radius = 0.03  # meters
metric = args.metric # see documentation of scipy.spatial.distance.pdist function for possible metrics

if args.down_sample:
    temporal_radius = 1  # samples
    tmin = 0.0
    twidth = 0.02 
    tmax = 0.8
    n_samples = int(round((tmax - tmin) / twidth))
else: # no downsampling - use a temporal radius nonetheless
    temporal_radius = 10  # samples
    tmin = -0.2
    twidth = 0.001
    tmax = 1.001 
    n_samples = int(round((tmax - tmin) / twidth))

# Load source space
src = mne.read_source_spaces(args.source)

# During inverse computation, the source space was downsampled (i.e. using ico4).
# Construct vertex-to-vertex distance matrices using only the vertices that
# are defined in the source solution.
dist = []
for hemi in [0, 1]:
    inuse = np.flatnonzero(src[0]['inuse'])
    dist.append(src[hemi]['dist'][np.ix_(inuse, inuse)].toarray())

# Load semantic norm data
m = loadmat(args.norms)
y = m['sorted']['mat'][0][0].astype(np.float)

# Some norm files have a typo
if 'wordsNoscand' in m['sorted'].dtype.names:
    stimuli = [x[0][0] for x in m['sorted']['wordsNoscand'][0][0]]
else:
    stimuli = [x[0][0] for x in m['sorted']['wordNoscand'][0][0]]

print 'Loading MEG data...'
n_stimuli = len(stimuli)
X = np.zeros((n_stimuli, 5124, n_samples))

if args.verbose: pbar = tqdm(total=n_stimuli, unit='stimuli')
for i, stimulus in enumerate(stimuli):
    # Search for the file
    options = [f for f in os.listdir(args.input_folder) if re.match(r'(^|.+-){stimulus}-lh.stc'.format(stimulus=stimulus), f)]
    if len(options) == 1:
        fname = os.path.join(args.input_folder, options[0][:-len('-lh.stc')])
        stc = mne.read_source_estimate(fname)
        if args.down_sample:
           # Downsample using the same procedure as for the data going into zero-shot
           stc = stc.bin(twidth, tstart=tmin, tstop=tmax, func=np.mean) 
        X[i] = stc.data
        if args.verbose: pbar.update(1)
    elif len(options) > 1:
        raise RuntimeError('Multiple files found for {stimulus}'.format(stimulus=stimulus))
    else:
        raise RuntimeError('Cannot find STC for {stimulus}'.format(stimulus=stimulus))
if args.verbose: pbar.close()

# Load source space
src = mne.read_source_spaces(args.source)

# During inverse computation, the source space was downsampled (i.e. using ico4).
# Construct vertex-to-vertex distance matrices using only the vertices that
# are defined in the source solution.
dist = []
if ('dist' in src[0]) == False:
    src = mne.add_source_space_distances(src, dist_limit=spatial_radius)

for hemi in [0, 1]:
    inuse = np.flatnonzero(src[0]['inuse'])
    dist.append(src[hemi]['dist'][np.ix_(stc.vertices[hemi], stc.vertices[hemi])].toarray())

# Load control variables if needed
if args.control is not None:
    print 'Controlling for given variables'
    m_c = loadmat(args.control)
    control = m_c['sorted']['mat'][0][0]

    # Make control variables be in the correct order
    control_words = [x[0][0] for x in m_c['sorted']['word'][0][0]]
    order = [control_words.index(w) for w in stimuli]
    control_words = [control_words[i] for i in order]
    control = control[order]
    control = control.astype(np.float)

    # Regress out control variables
    model = linear_model.LinearRegression()
    y -= model.fit(control, y).predict(control)
    X -= model.fit(control, X.reshape(X.shape[0], -1)).predict(control).reshape(X.shape)


def rsa_searchlight_step(x, dsm_y, temporal_radius):
    """Spatio-temporal RSA analysis (single step).

    This performs a single searchlight step of the RSA. This function is
    called in a parallel fashion across all searchlight steps.
    """
    results = []
    pvals = []
    n_stimuli = x.shape[0]
    for sample in range(temporal_radius, n_samples - temporal_radius):
        dsm_x = distance.pdist(
            x[:, :, sample - temporal_radius:sample + temporal_radius].reshape(n_stimuli, -1),
            metric='correlation'
        )
        r, p = stats.spearmanr(dsm_x, dsm_y)
        results.append(1 - r)
        pvals.append(p)
    return np.array(results), np.array(pvals)


def rsa_searchlight(X, y, spatial_radius=0.02, temporal_radius=10):
    """Compute spatio-temporal RSA using a searchlight pattern.

    Parameters
    ----------
    X : array (n_stimuli, n_features)
        The MEG data (where channels and samples are concatenated)
    y : array (n_stimuli, n_targets)
        The semantic norms
    spatial_radius : float
        The spatial radius of the searchlight in meters.
    temporal_radius: int
        The temporal radius of the searchlight in samples.

    Returns
    -------
    results : array (n_vertices, 2)
        A matrix where each row corresponds to a vertex in the source space.
        The two columns contain the RSA correlation value and the
        corresponding p-value.
    """
    # Compute the dissimilarity matrix of the semantic norms.
    dsm_y = distance.pdist(y, metric=metric)

    if not np.isfinite(dsm_y).all():
        raise RuntimeError('The DSM matrix of Y contains values that are NaN or Infinity.')

    # Progress bar
    if args.verbose: pbar = tqdm(total=len(dist[0]) + len(dist[1]))
    def progress(sequence):
        for item in sequence:
            if args.verbose: pbar.update(1)
            if args.break_after is not None and pbar.currval > args.break_after:
                break
            yield item

    # Use multiple cores to do the RSA computation
    parallel, my_rsa_searchlight_step, _ = mne.parallel.parallel_func(rsa_searchlight_step, args.jobs, verbose=False)
    results = []
    for hemi in [0, 1]:
        results += parallel(
            my_rsa_searchlight_step(
                X[:, np.flatnonzero(vertex_dist < spatial_radius) + hemi * 2562, :],
                dsm_y,
                temporal_radius
            )
            for vertex_dist in progress(dist[hemi])
        )
    if args.verbose: pbar.close()
    return np.array(results)


def save_rsa_results(results, filename, temporal_radius):
    """Save RSA results as two .stc files.

    Two .stc files are written, one with the correlation values and one with
    the p-values.

    Parameters
    ----------
    results : array (n_vertices, 2)
        The results produced by the `rsa_searchlight` function.
    filename : str
        The name of the .stc file to write the results to. P-values will be
        saved to filename + '-pvals'.
    temporal_radius : int
        The temporal radius of the searchlight. The function needs to know this
        in order to assign proper timestamps to the STC object.
    """
    stc = mne.SourceEstimate(
        data = 1-results[:, 0],
        vertices = [np.arange(2562), np.arange(2562)],
        tmin = tmin + 0.001 * temporal_radius,
        tstep = twidth,
        subject = 'fsaverage',
    )
    stc.save(filename)

    stc = mne.SourceEstimate(
        data = results[:, 1],
        vertices = [np.arange(2562), np.arange(2562)],
        tmin = tmin + 0.001 * temporal_radius,
        tstep = twidth,
        subject = 'fsaverage',
    )
    stc.save(filename + '-pvals')


print 'Computing RSA...'

# Whether to run abstract and concrete separately
if args.abscon:
    def return_indices_of_a(a, b):
      b_set = set(b)
      return [i for i, v in enumerate(a) if v in b_set] 

    abspath = os.path.dirname(args.norms)

    print 'Abstract words'

    fname = '/neuro/data/redness/stimuli/Redness1/inputWords2014-06-10_edited_abstractOnly.csv'
    if not os.path.exists(fname):
        fname = os.path.join(abspath, 'inputWords2014-06-10_edited_abstractOnly.csv')
    with open(fname) as f:
        abs_stimuli = [line.split(' ')[0] for line in f]
    indx = return_indices_of_a(stimuli, abs_stimuli)
    results = rsa_searchlight(X[indx], y[indx], spatial_radius, temporal_radius)
    save_rsa_results(results, args.output + '-abstract', temporal_radius)

    print 'Concrete words'

    fname = '/neuro/data/redness/stimuli/Redness1/inputWords2014-06-10_edited_concreteOnly.csv'
    if not os.path.exists(fname):
        fname = os.path.join(abspath, 'inputWords2014-06-10_edited_concreteOnly.csv')
    with open(fname) as f:
        con_stimuli = [line.split(' ')[0] for line in f]
    indx = return_indices_of_a(stimuli, con_stimuli)
    results = rsa_searchlight(X[indx], y[indx], spatial_radius, temporal_radius)
    save_rsa_results(results, args.output + '-concrete', temporal_radius)

else:
    results = rsa_searchlight(X, y, spatial_radius, temporal_radius)
    save_rsa_results(results, args.output, temporal_radius)
