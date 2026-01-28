import sys
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from tqdm import tqdm
from tifffile import imread

from csbdeep.utils import Path, normalize
from stardist import relabel_image_stardist, random_label_cmap, fill_label_holes, calculate_extents, \
    _draw_polygons

from stardist.matching import matching_dataset
from skimage.util import img_as_float32

lbl_cmap = random_label_cmap()

def show_reconstruction_acc(Y, branch, log_dir, results_dir):
    # Fitting ground-truth labels with star-convex polygons
    n_rays = [2**i for i in range(2,8)]
    scores = []
    for r in tqdm(n_rays):
        Y_reconstructed = [relabel_image_stardist(lbl, n_rays=r) for lbl in Y]
        mean_iou = matching_dataset(Y, Y_reconstructed, thresh=0, show_progress=False).mean_true_score
        scores.append(mean_iou)

    plt.figure(figsize=(8,5))
    plt.plot(n_rays, scores, 'o-')
    plt.xlabel('Number of rays for star-convex polygon')
    plt.ylabel('Reconstruction score (mean intersection over union)')
    plt.title("Accuracy of ground truth reconstruction (should be > 0.8 for a reasonable number of rays)")
    path = Path(results_dir+'/'+'/'+log_dir+'/plots/'+'n_rays_reconstruction_accuracy'+str(branch)+'.svg')
    plt.savefig(path)
    None;

def show_reconstruction_polygon(lbl, branch, log_dir,results_dir):
    n_rays = [2**i for i in range(2,8)]
    fig, ax = plt.subplots(2,3, figsize=(16,11))
    for a,r in zip(ax.flat,n_rays):
        a.imshow(relabel_image_stardist(lbl, n_rays=r), cmap=lbl_cmap)
        a.set_title('Reconstructed (%d rays)' % r)
        a.axis('off')
    plt.tight_layout();
    path = Path(results_dir+'/'+'/'+log_dir+'/plots/'+'n_rays_reconstruction_polygon'+str(branch)+'.svg')
    plt.savefig(path)

def load_and_preprocess_data(X, Y):
    X = list(map(imread,X))
    Y = list(map(imread,Y))
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,0,100, axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]
    return X, Y

def resize_to_target(image, target_dim):
    def make_divisible(x, divisor=16):
        return (x // divisor) * divisor
    
    h, w = image.shape[:2]
    scale = min(w / target_dim[0], h / target_dim[1])
    new_w = make_divisible(int(w / scale))
    new_h = make_divisible(int(h / scale))
    
    y_indices = np.linspace(0, h - 1, new_h).astype(int)
    x_indices = np.linspace(0, w - 1, new_w).astype(int)
    if image.ndim == 3:
        resized = image[y_indices[:, None], x_indices[None, :], :]
    else:
        resized = image[y_indices[:, None], x_indices[None, :]]
    return resized

def load_and_preprocess_data_hydra(X, Y1, Y2, target_dim=None):
    X = list(map(imread,X))
    Y1 = list(map(imread,Y1))
    Y2 = list(map(imread,Y2))

    if target_dim is not None:
        X = [resize_to_target(x, target_dim) for x in X]
        Y1 = [resize_to_target(y1, target_dim) for y1 in Y1]
        Y2 = [resize_to_target(y2, target_dim) for y2 in Y2]

    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,0,100, axis=axis_norm) for x in tqdm(X)]
    Y1 = [fill_label_holes(y1) for y1 in tqdm(Y1)]
    Y2 = [fill_label_holes(y2) for y2 in tqdm(Y2)]
    return X, Y1, Y2

def plot_img_label(img, lbl, img_title="image", lbl_title=["label"], **kwargs):
    fig, (ai,aj) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    aj.imshow(lbl, cmap=lbl_cmap)
    aj.set_title(lbl_title[0])
    plt.tight_layout()

def plot_img_label_hydra(img, lbl1, lbl2, img_title="image", lbl_title=["label1", "label2"], **kwargs):
    fig, (ai,aj,ak) = plt.subplots(1,3, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    aj.imshow(lbl1, cmap=lbl_cmap)
    aj.set_title(lbl_title[0])
    ak.imshow(lbl2, cmap=lbl_cmap)
    ak.set_title(lbl_title[1])
    plt.tight_layout()

def check_fov(Y, model):
    median_size = calculate_extents(list(Y), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if median_size.any() > fov.any():
        print("WARNING: median object size larger than field of view of the neural network.")

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    flipped = False
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
            flipped = True
    return img, mask, flipped

def random_fliprot_hydra(img, mask1, mask2): 
    assert img.ndim >= mask1.ndim == mask2.ndim
    axes = tuple(range(mask1.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask1.ndim, img.ndim))) 
    mask1 = mask1.transpose(perm) 
    mask2 = mask2.transpose(perm)
    flipped = False
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask1 = np.flip(mask1, axis=ax)
            mask2 = np.flip(mask2, axis=ax)
            flipped = True
    return img, mask1, mask2, flipped

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y, flipped = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

def augmenter_hydra(x, y1, y2):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y1, y2, flipped = random_fliprot_hydra(x, y1, y2)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y1, y2

def plot_metrics_vs_tau(stats,branch,log_dir, results_dir):
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend();

    path = Path(results_dir+'/'+log_dir+'/plots/'+'stats'+str(branch)+'_vs_tau.svg')
    plt.savefig(path)

def plot_individual_channel_predictions(img,labels):
    plt.figure(figsize=(8,8))
    plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    plt.axis('off');

def plot_individual_channel_predictions_hydra(img,labels):
    plt.figure(figsize=(8,8))
    plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
    plt.imshow(labels[0], cmap=lbl_cmap, alpha=0.5)
    plt.axis('off');

def example(model, X, Y, i, log_dir, results_dir, show_dist=True):
    axis_norm = (0,1)
    img = normalize(X[i], 0,100, axis=axis_norm)
    labels, details = model.predict_instances(img,prob_thresh=model.thresholds.prob, nms_thresh=model.thresholds.nms)

    plt.figure(figsize=(13,10))
    img_show = img if img.ndim==2 else img[...,0]
    coord, points, prob = details['coord'], details['points'], details['prob']
    plt.subplot(121); plt.imshow(img_show, cmap='gray'); plt.subplot(121).set_title('prediction'); plt.axis('off')
    a = plt.axis()
    _draw_polygons(coord, points, prob, show_dist=show_dist)
    plt.axis(a)
    plt.subplot(122); plt.imshow(img_show, cmap='gray'); plt.subplot(122).set_title('ground truth'); plt.axis('off')
    plt.imshow(Y[i], cmap=lbl_cmap, alpha=0.5)
    plt.tight_layout()
    path = Path(results_dir+'/'+log_dir+'/plots/'+'prediction_overlayed_'+str(i)+'.svg')
    plt.savefig(path)

def example_hydra(model, X, Y1, Y2, i, log_dir, results_dir, show_dist=True):
    axis_norm = (0,1)
    img = normalize(X[i], 0,100, axis=axis_norm)
    labels1, labels2 = model.predict_instances(img,prob_thresh1=model.thresholds1['prob'], prob_thresh2=model.thresholds2['prob'],
                            nms_thresh1=model.thresholds1['nms'], nms_thresh2=model.thresholds2['nms'])

    details1 = labels1[1]
    labels1 = labels1[0]

    details2 = labels2[1]
    labels2 = labels2[0]
    plt.figure(figsize=(13,10))
    img_show = img if img.ndim==2 else img[...,0]
    coord1, points1, prob1 = details1['coord'], details1['points'], details1['prob']
    coord2, points2, prob2 = details2['coord'], details2['points'], details2['prob']
    plt.subplot(131); plt.imshow(img_show, cmap='gray'); plt.subplot(131).set_title('prediction1'); plt.axis('off')
    a = plt.axis()
    _draw_polygons(coord1, points1, prob1, show_dist=show_dist)
    plt.axis(a)
    plt.subplot(132); plt.imshow(img_show, cmap='gray'); plt.subplot(132).set_title('prediction2'); plt.axis('off')
    b = plt.axis()
    _draw_polygons(coord2, points2, prob2, show_dist=show_dist)
    plt.axis(b)
    plt.subplot(133); plt.imshow(img_show, cmap='gray'); plt.subplot(133).set_title('ground truth'); plt.axis('off')
    plt.imshow(Y1[i], cmap=lbl_cmap, alpha=0.5)
    plt.imshow(Y2[i], cmap=lbl_cmap, alpha=0.5)
    plt.tight_layout()
    path = Path(results_dir+'/'+log_dir+'/plots/'+'prediction_overlayed_'+str(i)+'.svg')
    plt.savefig(path)

def create_assimilated_dict(l1,l2,l3,metric):
    assimilated_dict = dict()
    for d1, d2, d3 in zip(l1,l2,l3):
        assimilated_dict[str(d1['thresh'])] = dict()
        assimilated_dict[str(d1['thresh'])][0] = np.round(np.mean([d1[metric],d2[metric],d3[metric]]),4)
        assimilated_dict[str(d1['thresh'])][1] = np.round(np.std([d1[metric],d2[metric],d3[metric]]),4)

    return assimilated_dict