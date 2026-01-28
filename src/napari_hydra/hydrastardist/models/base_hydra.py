from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import sys
import warnings
import math
from tqdm import tqdm
from collections import namedtuple
from pathlib import Path
import threading
import functools
import scipy.ndimage as ndi
import numbers
from csbdeep.models.base_model import BaseModel
from napari_hydra.hydrastardist.utils.tf import export_SavedModel, keras_import, IS_TF_1, CARETensorBoard

import tensorflow as tf
K = keras_import('backend')
Sequence = keras_import('utils', 'Sequence')
Adam = keras_import('optimizers', 'Adam')
plot_model = keras_import('utils','plot_model')
ReduceLROnPlateau, TensorBoard = keras_import('callbacks', 'ReduceLROnPlateau', 'TensorBoard')

from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict, load_json, save_json
from csbdeep.internals.predict import tile_iterator, total_n_tiles
from csbdeep.internals.train import RollingSequence
from csbdeep.data import Resizer

from stardist.sample_patches import get_valid_inds
from stardist.nms import _ind_prob_thresh
from stardist.utils import _is_power_of_2,  _is_floatarray, optimize_threshold


def generic_masked_loss(mask, loss, weights=1, norm_by_mask=True, reg_weight=0, reg_penalty=K.abs):
    def _loss(y_true, y_pred):
        actual_loss = K.mean(mask * weights * loss(y_true, y_pred), axis=-1)
        norm_mask = (K.mean(mask) + K.epsilon()) if norm_by_mask else 1
        if reg_weight > 0:
            reg_loss = K.mean((1-mask) * reg_penalty(y_pred), axis=-1)
            return actual_loss / norm_mask + reg_weight * reg_loss
        else:
            return actual_loss / norm_mask
    return _loss

def masked_loss(mask, penalty, reg_weight, norm_by_mask):
    loss = lambda y_true, y_pred: penalty(y_true - y_pred)
    return generic_masked_loss(mask, loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_loss_mae(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.abs, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_loss_mse(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.square, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_metric_mae(mask):
    def relevant_mae(y_true, y_pred):
        return masked_loss(mask, K.abs, reg_weight=0, norm_by_mask=True)(y_true, y_pred)
    return relevant_mae

def masked_metric_mse(mask):
    def relevant_mse(y_true, y_pred):
        return masked_loss(mask, K.square, reg_weight=0, norm_by_mask=True)(y_true, y_pred)
    return relevant_mse

def kld(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.binary_crossentropy(y_true, y_pred) - K.binary_crossentropy(y_true, y_true), axis=-1)


def masked_loss_iou(mask, reg_weight=0, norm_by_mask=True):
    def iou_loss(y_true, y_pred):
        axis = -1 if backend_channels_last() else 1
        # y_pred can be negative (since not constrained) -> 'inter' can be very large for y_pred << 0
        # - clipping y_pred values at 0 can lead to vanishing gradients
        # - 'K.sign(y_pred)' term fixes issue by enforcing that y_pred values >= 0 always lead to larger 'inter' (lower loss)
        inter = K.mean(K.sign(y_pred)*K.square(K.minimum(y_true,y_pred)), axis=axis)
        union = K.mean(K.square(K.maximum(y_true,y_pred)), axis=axis)
        iou = inter/(union+K.epsilon())
        iou = K.expand_dims(iou,axis)
        loss = 1. - iou 
        return loss
    return generic_masked_loss(mask, iou_loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_metric_iou(mask, reg_weight=0, norm_by_mask=True):
    def iou_metric(y_true, y_pred):
        axis = -1 if backend_channels_last() else 1
        y_pred = K.maximum(0., y_pred)
        inter = K.mean(K.square(K.minimum(y_true,y_pred)), axis=axis)
        union = K.mean(K.square(K.maximum(y_true,y_pred)), axis=axis)
        iou = inter/(union+K.epsilon())
        loss = K.expand_dims(iou,axis)
        return loss
    return generic_masked_loss(mask, iou_metric, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_loss_wbr(mask, reg_weight=0, norm_by_mask=True):
    def wbr_loss(y_true, y_pred):
        y_true2 = y_true[1:]
        y_pred1 = y_pred[0]
        y_pred2 = y_pred[1:]
        #make all labels greater than 0 as 1, invert and then apply
        y_pred1 = tf.where(y_pred1>0,1.0,y_pred1)
        y_pred2 = tf.where(y_pred2>0,1.0,y_pred2)
        inv_y_pred2 = 1-y_pred2
        inv_y_true2 = 1-y_true2
        loss = 1/(1+K.epsilon()-(tf.reduce_sum(tf.multiply(inv_y_pred2,y_pred1))/tf.reduce_sum(inv_y_true2)))
        return loss
    return generic_masked_loss(mask, wbr_loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def weighted_categorical_crossentropy(weights, ndim):
    """ ndim = (2,3) """

    axis = -1 if backend_channels_last() else 1
    shape = [1]*(ndim+2)
    shape[axis] = len(weights)
    weights = np.broadcast_to(weights, shape)
    weights = K.constant(weights)

    def weighted_cce(y_true, y_pred):
        # ignore pixels that have y_true (prob_class) < 0
        mask = K.cast(y_true>=0, K.floatx())
        y_pred /= K.sum(y_pred+K.epsilon(), axis=axis, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        loss = - K.sum(weights*mask*y_true*K.log(y_pred), axis = axis)
        return loss

    return weighted_cce


class StarDistDataBase(RollingSequence):

    def __init__(self, X, Y1, Y2, n_rays, grid, batch_size, patch_size, length,
                 n_classes=None, classes=None,
                 use_gpu=True, sample_ind_cache=True, maxfilter_patch_size=None, augmenter=None, foreground_prob=0):

        super().__init__(data_size=len(X), batch_size=batch_size, length=length, shuffle=True)

        if isinstance(X, (np.ndarray, tuple, list)):
            X = [x.astype(np.float32, copy=False) for x in X]

        # sanity checks
        len(X)==len(Y1)==len(Y2) and len(X)>0 or _raise(ValueError("X and Y1 and Y2 can't be empty and must have same length"))

        if classes is None:
            # set classes to None for all images (i.e. defaults to every object instance assigned the same class)
            classes = (None,)*len(X)
        else:
            n_classes is not None or warnings.warn("Ignoring classes since n_classes is None")

        len(classes)==len(X) or _raise(ValueError("X and classes must have same length"))

        self.n_classes, self.classes = n_classes, classes

        nD = len(patch_size)
        assert nD in (2,3)
        x_ndim = X[0].ndim
        assert x_ndim in (nD,nD+1)

        if isinstance(X, (np.ndarray, tuple, list)) and \
           isinstance(Y1, (np.ndarray, tuple, list)) and \
           isinstance(Y2, (np.ndarray, tuple, list)):
            all(y1.ndim==nD and y2.ndim==nD and x.ndim==x_ndim and x.shape[:nD]==y1.shape==y2.shape for x,y1,y2 in zip(X,Y1,Y2)) or _raise(ValueError("images and masks should have corresponding shapes/dimensions"))
            all(x.shape[:nD]>=tuple(patch_size) for x in X) or _raise(ValueError("Some images are too small for given patch_size {patch_size}".format(patch_size=patch_size)))

        if x_ndim == nD:
            self.n_channel = None
        else:
            self.n_channel = X[0].shape[-1]
            if isinstance(X, (np.ndarray, tuple, list)):
                assert all(x.shape[-1]==self.n_channel for x in X)

        assert 0 <= foreground_prob <= 1

        self.X, self.Y1, self.Y2 = X, Y1, Y2
        self.n_rays = n_rays
        self.patch_size = patch_size
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in grid)
        self.grid = tuple(grid)
        self.use_gpu = bool(use_gpu)
        if augmenter is None:
            augmenter = lambda *args: args
        callable(augmenter) or _raise(ValueError("augmenter must be None or callable"))
        self.augmenter = augmenter
        self.foreground_prob = foreground_prob

        if self.use_gpu:
            from gputools import max_filter
            self.max_filter = lambda y, patch_size: max_filter(y.astype(np.float32), patch_size)
        else:
            from scipy.ndimage import maximum_filter
            self.max_filter = lambda y, patch_size: maximum_filter(y, patch_size, mode='constant')

        self.maxfilter_patch_size = maxfilter_patch_size if maxfilter_patch_size is not None else self.patch_size

        self.sample_ind_cache = sample_ind_cache
        self._ind_cache_fg  = {}
        self._ind_cache_all = {}
        self.lock = threading.Lock()


    def get_valid_inds(self, k, foreground_prob=None):
        if foreground_prob is None:
            foreground_prob = self.foreground_prob
        foreground_only = np.random.uniform() < foreground_prob
        _ind_cache = self._ind_cache_fg if foreground_only else self._ind_cache_all

        if k in _ind_cache:
            inds = _ind_cache[k]
        else:
            patch_filter = (lambda y,p: self.max_filter(y, self.maxfilter_patch_size) > 0) if foreground_only else None
            inds = get_valid_inds(self.Y1[k], self.patch_size, patch_filter=patch_filter)
            if self.sample_ind_cache:
                with self.lock:
                    _ind_cache[k] = inds

        if foreground_only and len(inds[0])==0:
            # no foreground pixels available
            return self.get_valid_inds(k, foreground_prob=0)
        return inds


    def channels_as_tuple(self, x):
        if self.n_channel is None:
            return (x,)
        else:
            return tuple(x[...,i] for i in range(self.n_channel))



class StarDistBase(BaseModel):

    def __init__(self, config, name=None, basedir='.'):
        super().__init__(config=config, name=name, basedir=basedir)
        threshs1 = dict(prob=None, nms=None)
        threshs2 = dict(prob=None, nms=None)

        if basedir is not None:
            try:
                threshs1 = load_json(str(self.logdir / 'thresholds1.json'))
                print("Loading thresholds from 'thresholds1.json'.")
                if threshs1.get('prob') is None or not (0 < threshs1.get('prob') < 1):
                    print("- Invalid 'prob' threshold (%s), using default value." % str(threshs1.get('prob')))
                    threshs1['prob'] = None
                if threshs1.get('nms') is None or not (0 < threshs1.get('nms') < 1):
                    print("- Invalid 'nms' threshold (%s), using default value." % str(threshs1.get('nms')))
                    threshs1['nms'] = None
            except FileNotFoundError:
                if config is None and len(tuple(self.logdir.glob('*.h5'))) > 0:
                    print("Couldn't load thresholds from 'thresholds1.json', using default values. "
                          "(Call 'optimize_thresholds' to change that.)")

            try:
                threshs2 = load_json(str(self.logdir / 'thresholds2.json'))
                print("Loading thresholds from 'thresholds2.json'.")
                if threshs2.get('prob') is None or not (0 < threshs2.get('prob') < 1):
                    print("- Invalid 'prob' threshold (%s), using default value." % str(threshs2.get('prob')))
                    threshs2['prob'] = None
                if threshs2.get('nms') is None or not (0 < threshs2.get('nms') < 1):
                    print("- Invalid 'nms' threshold (%s), using default value." % str(threshs2.get('nms')))
                    threshs2['nms'] = None
            except FileNotFoundError:
                if config is None and len(tuple(self.logdir.glob('*.h5'))) > 0:
                    print("Couldn't load thresholds from 'thresholds2.json', using default values. "
                          "(Call 'optimize_thresholds' to change that.)")

        self.thresholds1 = dict (
            prob = 0.5 if threshs1['prob'] is None else threshs1['prob'],
            nms  = 0.4 if threshs1['nms']  is None else threshs1['nms'],
        )
        print("Using default values: prob_thresh1={prob:g}, nms_thresh1={nms:g}.".format(prob=self.thresholds1['prob'], nms=self.thresholds1['nms']))

        self.thresholds2 = dict (
            prob = 0.5 if threshs2['prob'] is None else threshs2['prob'],
            nms  = 0.4 if threshs2['nms']  is None else threshs2['nms'],
        )
        print("Using default values: prob_thresh2={prob:g}, nms_thresh2={nms:g}.".format(prob=self.thresholds2['prob'], nms=self.thresholds2['nms']))

    @property
    def thresholds(self):
        return self._thresholds

    def _is_multiclass(self):
        return (self.config.n_classes is not None)

    def _parse_classes_arg(self, classes, length):
        """ creates a proper classes tuple from different possible "classes" arguments in model.train()

        classes can be
          "auto" -> all objects will be assigned to the first foreground class (unless n_classes is None)
          single integer -> all objects will be assigned that class
          tuple, list, ndarray -> do nothing (needs to be of given length)

        returns a tuple of given length
        """
        if isinstance(classes, str):
            classes == "auto" or _raise(ValueError(f"classes = '{classes}': only 'auto' supported as string argument for classes"))
            if self.config.n_classes is None:
                classes = None
            elif self.config.n_classes == 1:
                classes = (1,)*length
            else:
                raise ValueError("using classes = 'auto' for n_classes > 1 not supported")
        elif isinstance(classes, (tuple, list, np.ndarray)):
            len(classes) == length or _raise(ValueError(f"len(classes) should be {length}!"))
        else:
            raise ValueError("classes should either be 'auto' or a list of scalars/label dicts")
        return classes

    @thresholds.setter
    def thresholds(self, d):
        self._thresholds = namedtuple('Thresholds',d.keys())(*d.values())


    def prepare_for_training(self, optimizer=None):
        """Prepare for neural network training.

        Compiles the model and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.

        """
        if optimizer is None:
            optimizer = Adam(self.config.train_learning_rate)

        masked_dist_loss = {'mse': masked_loss_mse,
                            'mae': masked_loss_mae,
                            'iou': masked_loss_iou,
                            }[self.config.train_dist_loss]
        prob1_loss = prob2_loss = 'binary_crossentropy'


        def split_dist_true_mask(dist_true_mask):
            return tf.split(dist_true_mask, num_or_size_splits=[self.config.n_rays,-1], axis=-1)

        def dist_loss(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_dist_loss(dist_mask, reg_weight=self.config.train_background_reg)(dist_true, dist_pred)

        def dist_iou_metric(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_metric_iou(dist_mask, reg_weight=0)(dist_true, dist_pred)

        def relevant_mae(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_metric_mae(dist_mask)(dist_true, dist_pred)

        def relevant_mse(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_metric_mse(dist_mask)(dist_true, dist_pred)


        if self._is_multiclass():
            prob_class1_loss = prob_class2_loss = weighted_categorical_crossentropy(self.config.train_class_weights, ndim=self.config.n_dim)
            loss = [[prob1_loss, dist_loss, prob_class1_loss], [prob2_loss, dist_loss, prob_class2_loss]]
        else:
            loss = [[prob1_loss, dist_loss], [prob2_loss, dist_loss]]


        self.keras_model.compile(optimizer, loss         = {'prob1':loss[0][0],
                                                            'dist1':loss[0][1],
                                                            'prob2':loss[1][0],
                                                            'dist2':loss[1][1]},
                                            loss_weights = {'prob1':list(self.config.train_loss_weights)[0][0],
                                                            'dist1':list(self.config.train_loss_weights)[0][1],
                                                            'prob2':list(self.config.train_loss_weights)[1][0],
                                                            'dist2':list(self.config.train_loss_weights)[1][1]},
                                            metrics      = {'prob1': kld,
                                                            'dist1': [relevant_mae, relevant_mse, dist_iou_metric],
                                                            'prob2': kld,
                                                            'dist2': [relevant_mae, relevant_mse, dist_iou_metric]})

        self.callbacks = []
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                if IS_TF_1:
                    self.callbacks.append(CARETensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=False))
                else:
                    self.callbacks.append(TensorBoard(log_dir=str(self.logdir/'logs'), write_graph=False, profile_batch=0))

        if self.config.train_reduce_lr is not None:
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            # TF2: add as first callback to put 'lr' in the logs for TensorBoard
            self.callbacks.insert(0,ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def _predict_setup(self, img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs):
        """ Shared setup code between `predict` and `predict_sparse` """
        if n_tiles is None:
            n_tiles = [1]*img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)
        all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))

        n_tiles = tuple(map(int,n_tiles))

        axes     = self._normalize_axes(img, axes)
        axes_net = self.config.axes

        _permute_axes = self._make_permute_axes(axes, axes_net)
        x = _permute_axes(img) # x has axes_net semantics

        channel = axes_dict(axes_net)['C']
        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())
        axes_net_div_by = self._axes_div_by(axes_net)

        grid = tuple(self.config.grid)
        len(grid) == len(axes_net)-1 or _raise(ValueError())
        grid_dict = dict(zip(axes_net.replace('C',''),grid))

        normalizer = self._check_normalizer_resizer(normalizer, None)[0]
        resizer = StarDistPadAndCropResizer(grid=grid_dict)

        x = normalizer.before(x, axes_net)
        x = resizer.before(x, axes_net, axes_net_div_by)

        if not _is_floatarray(x):
            warnings.warn("Predicting on non-float input... ( forgot to normalize? )")

        def predict_direct(x):
            output_dict = self.keras_model.predict(x[np.newaxis], **predict_kwargs)
            ys1 = (output_dict['prob1'], output_dict['dist1'])
            ys2 = (output_dict['prob2'], output_dict['dist2'])
            return tuple(y[0] for y in ys1), tuple(y[0] for y in ys2)

        def tiling_setup():
            assert np.prod(n_tiles) > 1
            tiling_axes   = axes_net.replace('C','') # axes eligible for tiling
            x_tiling_axis = tuple(axes_dict(axes_net)[a] for a in tiling_axes) # numerical axis ids for x
            axes_net_tile_overlaps, _ = self._axes_tile_overlap(axes_net)
            # hack: permute tiling axis in the same way as img -> x was permuted
            _n_tiles = _permute_axes(np.empty(n_tiles,bool)).shape
            (all(_n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis) or
                _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))

            sh = [s//grid_dict.get(a,1) for a,s in zip(axes_net,x.shape)]
            sh[channel] = None
            def create_empty_output(n_channel, dtype=np.float32):
                sh[channel] = n_channel
                return np.empty(sh,dtype)

            if callable(show_tile_progress):
                progress, _show_tile_progress = show_tile_progress, True
            else:
                progress, _show_tile_progress = tqdm, show_tile_progress

            n_block_overlaps = [int(np.ceil(overlap/blocksize)) for overlap, blocksize
                                in zip(axes_net_tile_overlaps, axes_net_div_by)]

            num_tiles_used = total_n_tiles(x, _n_tiles, block_sizes=axes_net_div_by, n_block_overlaps=n_block_overlaps)

            tile_generator = progress(tile_iterator(x, _n_tiles, block_sizes=axes_net_div_by, n_block_overlaps=n_block_overlaps),
                                                    disable=(not _show_tile_progress), total=num_tiles_used)

            return tile_generator, tuple(sh), create_empty_output

        return x, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles, grid, grid_dict, channel, predict_direct, tiling_setup


    def _predict_generator(self, img, axes=None, normalizer=None, n_tiles=None, show_tile_progress=True, **predict_kwargs):
        """Predict.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool or callable
            If boolean, indicates whether to show progress (via tqdm) during tiled prediction.
            If callable, must be a drop-in replacement for tqdm.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.

        Returns
        -------
        (:class:`numpy.ndarray`, :class:`numpy.ndarray`, [:class:`numpy.ndarray`])
            Returns the tuple (`prob`, `dist`, [`prob_class`]) of per-pixel object probabilities and star-convex polygon/polyhedra distances.
            In multiclass prediction mode, `prob_class` is the probability map for each of the 1+'n_classes' classes (first class is background)

        """

        predict_kwargs.setdefault('verbose', 0)
        x, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles, grid, grid_dict, channel, predict_direct, tiling_setup = \
            self._predict_setup(img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs)

        if np.prod(n_tiles) > 1:
            tile_generator, output_shape, create_empty_output = tiling_setup()

            prob1 = create_empty_output(1)
            dist1 = create_empty_output(self.config.n_rays)

            prob2 = create_empty_output(1)
            dist2 = create_empty_output(self.config.n_rays)
            if self._is_multiclass():
                prob_class1 = create_empty_output(self.config.n_classes+1)
                prob_class2 = create_empty_output(self.config.n_classes+1)
                result1 = (prob1, dist1, prob_class1)
                result2 = (prob2, dist2, prob_class2)
            else:
                result1 = (prob1, dist1)
                result2 = (prob2, dist2)

            for tile, s_src, s_dst in tile_generator:
                # predict_direct -> prob, dist, [prob_class if multi_class]
                result_tile1, result_tile2 = predict_direct(tile)
                # account for grid
                s_src = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_src,axes_net)]
                s_dst = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_dst,axes_net)]
                # prob and dist have different channel dimensionality than image x
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)
                for part, part_tile in zip(result1, result_tile1):
                    part[s_dst] = part_tile[s_src]
                for part, part_tile in zip(result2, result_tile2):
                    part[s_dst] = part_tile[s_src]
                yield  # yield None after each processed tile
        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            result1, result2 = predict_direct(x)

        result1 = [resizer.after(part, axes_net) for part in result1]
        result2 = [resizer.after(part, axes_net) for part in result2]
        # result = (prob, dist) for legacy or (prob, dist, prob_class) for multiclass

        # prob
        result1[0] = np.take(result1[0],0,axis=channel)
        # dist
        result1[1] = np.maximum(1e-3, result1[1]) # avoid small dist values to prevent problems with Qhull
        result1[1] = np.moveaxis(result1[1],channel,-1)

        # prob
        result2[0] = np.take(result2[0],0,axis=channel)
        # dist
        result2[1] = np.maximum(1e-3, result2[1]) # avoid small dist values to prevent problems with Qhull
        result2[1] = np.moveaxis(result2[1],channel,-1)

        if self._is_multiclass():
            # prob_class
            result1[2] = np.moveaxis(result1[2],channel,-1)
            result2[2] = np.moveaxis(result2[2],channel,-1)
        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        yield tuple(result1), tuple(result2)


    @functools.wraps(_predict_generator)
    def predict(self, *args, **kwargs):
        # return last "yield"ed value of generator
        r = None
        for r in self._predict_generator(*args, **kwargs):
            pass
        return r


    def _predict_sparse_generator(self, img, prob_thresh1=None, prob_thresh2=None, axes=None, normalizer=None, n_tiles=None, show_tile_progress=True, b=2, **predict_kwargs):
        """ Sparse version of model.predict()
        Returns
        -------
        (prob, dist, [prob_class], points)   flat list of probs, dists, (optional prob_class) and points
        """
        if prob_thresh1 is None: 
            prob_thresh1 = self.thresholds1['prob']

        if prob_thresh2 is None: 
            prob_thresh2 = self.thresholds2['prob']

        predict_kwargs.setdefault('verbose', 0)
        x, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles, grid, grid_dict, channel, predict_direct, tiling_setup = \
            self._predict_setup(img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs)

        def _prep(prob, dist):
            prob = np.take(prob,0,axis=channel)
            dist = np.moveaxis(dist,channel,-1)
            dist = np.maximum(1e-3, dist)
            return prob, dist

        prob1a, dist1a, points1a, prob_class1a = [],[],[], []
        prob2a, dist2a, points2a, prob_class2a = [],[],[], []

        if np.prod(n_tiles) > 1:
            tile_generator, output_shape, create_empty_output = tiling_setup()

            sh1 = list(output_shape)
            sh1[channel] = 1;

            sh2 = list(output_shape)
            sh2[channel] = 1;

            prob1a, dist1a, points1a, prob_class1 = [],[],[], []
            prob2a, dist2a, points2a, prob_class2 = [],[],[], []

            for tile, s_src, s_dst in tile_generator:

                results_tile1, results_tile2 = predict_direct(tile)

                # account for grid
                s_src = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_src,axes_net)]
                s_dst = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_dst,axes_net)]
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)

                prob_tile1, dist_tile1 = results_tile1[:2]
                prob_tile1, dist_tile1 = _prep(prob_tile1[s_src], dist_tile1[s_src])

                prob_tile2, dist_tile2 = results_tile2[:2]
                prob_tile2, dist_tile2 = _prep(prob_tile2[s_src], dist_tile2[s_src])

                bs1 = list((b if s.start==0 else -1, b if s.stop==_sh else -1) for s,_sh in zip(s_dst, sh1))
                bs1.pop(channel)
                inds1   = _ind_prob_thresh(prob_tile1, prob_thresh1, b=bs1)
                prob1a.extend(prob_tile1[inds1].copy())
                dist1a.extend(dist_tile1[inds1].copy())
                _points1 = np.stack(np.where(inds1), axis=1)

                bs2 = list((b if s.start==0 else -1, b if s.stop==_sh else -1) for s,_sh in zip(s_dst, sh2))
                bs2.pop(channel)
                inds2   = _ind_prob_thresh(prob_tile2, prob_thresh2, b=bs2)
                prob2a.extend(prob_tile2[inds2].copy())
                dist2a.extend(dist_tile2[inds2].copy())
                _points2 = np.stack(np.where(inds2), axis=1)

                offset = list(s.start for i,s in enumerate(s_dst))
                offset.pop(channel)

                _points1 = _points1 + np.array(offset).reshape((1,len(offset)))
                _points1 = _points1 * np.array(self.config.grid).reshape((1,len(self.config.grid)))
                points1a.extend(_points1)

                _points2 = _points2 + np.array(offset).reshape((1,len(offset)))
                _points2 = _points2 * np.array(self.config.grid).reshape((1,len(self.config.grid)))
                points2a.extend(_points2)

                if self._is_multiclass():
                    p1 = results_tile1[2][s_src].copy()
                    p1 = np.moveaxis(p1,channel,-1)
                    prob_class1a.extend(p1[inds1])

                    p2 = results_tile2[2][s_src].copy()
                    p2 = np.moveaxis(p2,channel,-1)
                    prob_class1a.extend(p2[inds2])
                yield  # yield None after each processed tile

        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            results1, results2 = predict_direct(x)
            prob1, dist1 = results1[:2]
            prob1, dist1 = _prep(prob1, dist1)
            inds1   = _ind_prob_thresh(prob1, prob_thresh1, b=b)
            prob1a = prob1[inds1].copy()
            dist1a = dist1[inds1].copy()
            _points1 = np.stack(np.where(inds1), axis=1)
            points1a = (_points1 * np.array(self.config.grid).reshape((1,len(self.config.grid))))

            prob2, dist2 = results2[:2]
            prob2, dist2 = _prep(prob2, dist2)
            inds2   = _ind_prob_thresh(prob2, prob_thresh2, b=b)
            prob2a = prob2[inds2].copy()
            dist2a = dist2[inds2].copy()
            _points2 = np.stack(np.where(inds2), axis=1)
            points2a = (_points2 * np.array(self.config.grid).reshape((1,len(self.config.grid))))

            if self._is_multiclass():
                p1 = np.moveaxis(results1[2],channel,-1)
                prob_class1a = p1[inds1].copy()

                p2 = np.moveaxis(results2[2],channel,-1)
                prob_class2a = p2[inds2].copy()


        prob1a = np.asarray(prob1a)
        dist1a = np.asarray(dist1a).reshape((-1,self.config.n_rays))
        points1a = np.asarray(points1a).reshape((-1,self.config.n_dim))

        prob2a = np.asarray(prob2a)
        dist2a = np.asarray(dist2a).reshape((-1,self.config.n_rays))
        points2a = np.asarray(points2a).reshape((-1,self.config.n_dim))

        idx1 = resizer.filter_points(x.ndim, points1a, axes_net)
        prob1a = prob1a[idx1]
        dist1a = dist1a[idx1]
        points1a = points1a[idx1]

        idx2 = resizer.filter_points(x.ndim, points2a, axes_net)
        prob2a = prob2a[idx2]
        dist2a = dist2a[idx2]
        points2a = points2a[idx2]

        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        if self._is_multiclass():
            prob_class1a = np.asarray(prob_class1a).reshape((-1,self.config.n_classes+1))
            prob_class1a = prob_class1a[idx1]

            prob_class2a = np.asarray(prob_class2a).reshape((-1,self.config.n_classes+1))
            prob_class2a = prob_class2a[idx2]
            yield (prob1a, dist1a, prob_class1a, points1a), (prob2a, dist2a, prob_class2a, points2a)
        else:
            prob_class1a = None

            prob_class2a = None
            yield (prob1a, dist1a, points1a), (prob2a, dist2a, points2a)


    @functools.wraps(_predict_sparse_generator)
    def predict_sparse(self, *args, **kwargs):
        # return last "yield"ed value of generator
        r = None
        for r in self._predict_sparse_generator(*args, **kwargs):
            pass
        return r


    def _predict_instances_generator(self, img, axes=None, normalizer=None,
                                     sparse=True,
                                     prob_thresh1=None, nms_thresh1=None,
                                     prob_thresh2=None, nms_thresh2=None,
                                     scale=None,
                                     n_tiles=None, show_tile_progress=True,
                                     verbose=False,
                                     return_labels=True,
                                     predict_kwargs=None, nms_kwargs=None,
                                     overlap_label=None, return_predict=False):
        """Predict instance segmentation from input image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        sparse: bool
            If true, aggregate probabilities/distances sparsely during tiled
            prediction to save memory (recommended).
        prob_thresh : float or None
            Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : float or None
            Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        scale: None or float or iterable
            Scale the input image internally by this factor and rescale the output accordingly.
            All spatial axes (X,Y,Z) will be scaled if a scalar value is provided.
            Alternatively, multiple scale values (compatible with input `axes`) can be used
            for more fine-grained control (scale values for non-spatial axes must be 1).
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        verbose: bool
            Whether to print some info messages.
        return_labels: bool
            Whether to create a label image, otherwise return None in its place.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.
        nms_kwargs: dict
            Keyword arguments for non-maximum suppression.
        overlap_label: scalar or None
            if not None, label the regions where polygons overlap with that value
        return_predict: bool
            Also return the outputs of :func:`predict` (in a separate tuple)
            If True, implies sparse = False

        Returns
        -------
        (:class:`numpy.ndarray`, dict), (optional: return tuple of :func:`predict`)
            Returns a tuple of the label instances image and also
            a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.

        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if nms_kwargs is None:
            nms_kwargs = {}

        if return_predict and sparse:
            sparse = False
            warnings.warn("Setting sparse to False because return_predict is True")

        nms_kwargs.setdefault("verbose", verbose)

        _axes         = self._normalize_axes(img, axes)
        _axes_net     = self.config.axes
        _permute_axes = self._make_permute_axes(_axes, _axes_net)
        _shape_inst   = tuple(s for s,a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')

        if scale is not None:
            if isinstance(scale, numbers.Number):
                scale = tuple(scale if a in 'XYZ' else 1 for a in _axes)
            scale = tuple(scale)
            len(scale) == len(_axes) or _raise(ValueError(f"scale {scale} must be of length {len(_axes)}, i.e. one value for each of the axes {_axes}"))
            for s,a in zip(scale,_axes):
                s > 0 or _raise(ValueError("scale values must be greater than 0"))
                (s in (1,None) or a in 'XYZ') or warnings.warn(f"replacing scale value {s} for non-spatial axis {a} with 1")
            scale = tuple(s if a in 'XYZ' else 1 for s,a in zip(scale,_axes))
            verbose and print(f"scaling image by factors {scale} for axes {_axes}")
            img = ndi.zoom(img, scale, order=1)

        yield 'predict'  # indicate that prediction is starting
        res = None
        if sparse:
            for temp in self._predict_sparse_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                                      prob_thresh1=prob_thresh1, prob_thresh2=prob_thresh2, 
                                                      show_tile_progress=show_tile_progress, **predict_kwargs):
                if temp == None:
                    res1 = None
                    res2 = None
                else:
                    res1 = temp[0]
                    res2 = temp[1]
                if res1 is None and res2 is None:
                    yield 'tile'  # yield 'tile' each time a tile has been processed
        else:
            for temp in self._predict_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                               show_tile_progress=show_tile_progress, **predict_kwargs):
                if temp == None:
                    res1 = None
                    res2 = None
                else:
                    res1 = temp[0]
                    res2 = temp[1]
                if res1 is None and res2:
                    yield 'tile'  # yield 'tile' each time a tile has been processed

            res1 = res1 + (None,)
            res2 = res2 + (None,)

        if self._is_multiclass():
            prob1, dist1, prob_class1, points1 = res1
            prob2, dist2, prob_class2, points2 = res2
        else:
            prob1, dist1, points1 = res1
            prob_class1 = None
            prob2, dist2, points2 = res2
            prob_class2 = None

        yield 'nms'  # indicate that non-maximum suppression is starting
        res_instances1 = self._instances_from_prediction(_shape_inst, prob1, dist1,
                                                        points=points1,
                                                        prob_class=prob_class1,
                                                        prob_thresh=prob_thresh1,
                                                        nms_thresh=nms_thresh1,
                                                        scale=(None if scale is None else dict(zip(_axes,scale))),
                                                        return_labels=return_labels,
                                                        overlap_label=overlap_label,
                                                        **nms_kwargs)

        res_instances2 = self._instances_from_prediction(_shape_inst, prob2, dist2,
                                                        points=points2,
                                                        prob_class=prob_class2,
                                                        prob_thresh=prob_thresh2,
                                                        nms_thresh=nms_thresh2,
                                                        scale=(None if scale is None else dict(zip(_axes,scale))),
                                                        return_labels=return_labels,
                                                        overlap_label=overlap_label,
                                                        **nms_kwargs)

        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        if return_predict:
            yield (res_instances1, tuple(res1[:-1])), (res_instances2, tuple(res2[:-1]))
        else:
            yield res_instances1, res_instances2


    @functools.wraps(_predict_instances_generator)
    def predict_instances(self, *args, **kwargs):
        # the reason why the actual computation happens as a generator function
        # (in '_predict_instances_generator') is that the generator is called
        # from the stardist napari plugin, which has its benefits regarding
        # control flow and progress display. however, typical use cases should
        # almost always use this function ('predict_instances'), and shouldn't
        # even notice (thanks to @functools.wraps) that it wraps the generator
        # function. note that similar reasoning applies to 'predict' and
        # 'predict_sparse'.

        # return last "yield"ed value of generator
        r = None
        for r in self._predict_instances_generator(*args, **kwargs):
            pass
        return r

    def predict_instances_big(self, img, axes, block_size, min_overlap, context=None,
                              labels_out1=None, labels_out2=None, labels_out_dtype=np.int32, show_progress=True, **kwargs):
        """Predict instance segmentation from very large input images.

        Intended to be used when `predict_instances` cannot be used due to memory limitations.
        This function will break the input image into blocks and process them individually
        via `predict_instances` and assemble all the partial results. If used as intended, the result
        should be the same as if `predict_instances` was used directly on the whole image.

        **Important**: The crucial assumption is that all predicted object instances are smaller than
                       the provided `min_overlap`. Also, it must hold that: min_overlap + 2*context < block_size.

        Example
        -------
        >>> img.shape
        (20000, 20000)
        >>> labels, polys = model.predict_instances_big(img, axes='YX', block_size=4096,
                                                        min_overlap=128, context=128, n_tiles=(4,4))

        Parameters
        ----------
        img: :class:`numpy.ndarray` or similar
            Input image
        axes: str
            Axes of the input ``img`` (such as 'YX', 'ZYX', 'YXC', etc.)
        block_size: int or iterable of int
            Process input image in blocks of the provided shape.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        min_overlap: int or iterable of int
            Amount of guaranteed overlap between blocks.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        context: int or iterable of int, or None
            Amount of image context on all sides of a block, which is discarded.
            If None, uses an automatic estimate that should work in many cases.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        labels_out: :class:`numpy.ndarray` or similar, or None, or False
            numpy array or similar (must be of correct shape) to which the label image is written.
            If None, will allocate a numpy array of the correct shape and data type ``labels_out_dtype``.
            If False, will not write the label image (useful if only the dictionary is needed).
        labels_out_dtype: str or dtype
            Data type of returned label image if ``labels_out=None`` (has no effect otherwise).
        show_progress: bool
            Show progress bar for block processing.
        kwargs: dict
            Keyword arguments for ``predict_instances``.

        Returns
        -------
        (:class:`numpy.ndarray` or False, dict)
            Returns the label image and a dictionary with the details (coordinates, etc.) of the polygons/polyhedra.

        """
        from stardist.big import _grid_divisible, BlockND, OBJECT_KEYS
        from stardist.matching import relabel_sequential

        n = img.ndim
        axes = axes_check_and_normalize(axes, length=n)
        grid = self._axes_div_by(axes)
        axes_out = self._axes_out.replace('C','')
        shape_dict = dict(zip(axes,img.shape))
        shape_out = tuple(shape_dict[a] for a in axes_out)

        if context is None:
            context = self._axes_tile_overlap(axes)

        if np.isscalar(block_size):  block_size  = n*[block_size]
        if np.isscalar(min_overlap): min_overlap = n*[min_overlap]
        if np.isscalar(context):     context     = n*[context]
        block_size, min_overlap, context = list(block_size), list(min_overlap), list(context)
        assert n == len(block_size) == len(min_overlap) == len(context)

        if 'C' in axes:
            # single block for channel axis
            i = axes_dict(axes)['C']
            
            block_size[i] = img.shape[i]
            min_overlap[i] = context[i] = 0

        block_size  = tuple(_grid_divisible(g, v, name='block_size',  verbose=False) for v,g,a in zip(block_size, grid,axes))
        min_overlap = tuple(_grid_divisible(g, v, name='min_overlap', verbose=False) for v,g,a in zip(min_overlap,grid,axes))
        context     = tuple(_grid_divisible(g, v, name='context',     verbose=False) for v,g,a in zip(context,    grid,axes))

        print(f'effective: block_size={block_size}, min_overlap={min_overlap}, context={context}', flush=True)

        for a,c,o in zip(axes,context,self._axes_tile_overlap(axes)):
            if c < o:
                print(f"{a}: context of {c} is small, recommended to use at least {o}", flush=True)

        # create block cover
        blocks = BlockND.cover(img.shape, axes, block_size, min_overlap, context, grid)

        if np.isscalar(labels_out1) and bool(labels_out1) is False and np.isscalar(labels_out1) and \
        bool(labels_out2) is False :
            labels_out1 = None
            labels_out2 = None
        else:
            if labels_out1 is None and labels_out2 is None:
                labels_out1 = np.zeros(shape_out, dtype=labels_out_dtype)
                labels_out2 = np.zeros(shape_out, dtype=labels_out_dtype)
            else:
                labels_out1.shape == shape_out or _raise(ValueError(f"'labels_out1' must have shape {shape_out} (axes {axes_out})."))
                labels_out2.shape == shape_out or _raise(ValueError(f"'labels_out2' must have shape {shape_out} (axes {axes_out})."))

        polys1_all = {}
        polys2_all = {}

        label_offset1 = 1
        label_offset2 = 1

        kwargs_override = dict(axes=axes, overlap_label=None, return_labels=True, return_predict=False)
        if show_progress:
            kwargs_override['show_tile_progress'] = False # disable progress for predict_instances
        for k,v in kwargs_override.items():
            if k in kwargs: print(f"changing '{k}' from {kwargs[k]} to {v}", flush=True)
            kwargs[k] = v

        blocks = tqdm(blocks, disable=(not show_progress))
        # actual computation
        for block in blocks:
            (labels1, polys1), (labels2, polys2) = self.predict_instances(block.read(img, axes=axes), **kwargs)
            labels1 = block.crop_context(labels1, axes=axes_out)
            labels1, polys1 = block.filter_objects(labels1, polys1, axes=axes_out)

            labels2 = block.crop_context(labels2, axes=axes_out)
            labels2, polys2 = block.filter_objects(labels2, polys2, axes=axes_out)
            
            labels1 = relabel_sequential(labels1, label_offset1)[0]
            labels2 = relabel_sequential(labels2, label_offset2)[0]

            if labels_out1 is not None and labels_out2 is not None:
                block.write(labels_out1, labels2, axes=axes_out)
                block.write(labels_out2, labels2, axes=axes_out)

            for k,v in polys1.items():
                polys1_all.setdefault(k,[]).append(v)

            for k,v in polys2.items():
                polys2_all.setdefault(k,[]).append(v)

            label_offset1 += len(polys1['prob'])
            label_offset2 += len(polys2['prob'])
            del labels1, labels2

        polys1_all = {k: (np.concatenate(v) if k in OBJECT_KEYS else v[0]) for k,v in polys1_all.items()}
        polys2_all = {k: (np.concatenate(v) if k in OBJECT_KEYS else v[0]) for k,v in polys2_all.items()}

        return (labels_out1, polys1_all), (labels_out2, polys2_all)


    def optimize_thresholds(self, X_val, Y1_val, Y2_val, nms_threshs=[0.3,0.4,0.5], iou_threshs=[0.3,0.5,0.7], predict_kwargs=None, optimize_kwargs=None, save_to_json=True):

        """Optimize two thresholds (probability, NMS overlap) necessary for predicting object instances.

        Note that the default thresholds yield good results in many cases, but optimizing
        the thresholds for a particular dataset can further improve performance.

        The optimized thresholds are automatically used for all further predictions
        and also written to the model directory.

        See ``utils.optimize_threshold`` for details and possible choices for ``optimize_kwargs``.

        Parameters
        ----------
        X_val : list of ndarray
            (Validation) input images (must be normalized) to use for threshold tuning.
        Y_val : list of ndarray
            (Validation) label images to use for threshold tuning.
        nms_threshs : list of float
            List of overlap thresholds to be considered for NMS.
            For each value in this list, optimization is run to find a corresponding prob_thresh value.
        iou_threshs : list of float
            List of intersection over union (IOU) thresholds for which
            the (average) matching performance is considered to tune the thresholds.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of this class.
            (If not provided, will guess value for `n_tiles` to prevent out of memory errors.)
        optimize_kwargs: dict
            Keyword arguments for ``utils.optimize_threshold`` function.

        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if optimize_kwargs is None:
            optimize_kwargs = {}

        def _predict_kwargs(x):
            if 'n_tiles' in predict_kwargs:
                return predict_kwargs
            else:
                return {**predict_kwargs, 'n_tiles': self._guess_n_tiles(x), 'show_tile_progress': False}

        # only take first two elements of predict in case multi class is activated
        temp = [self.predict(x, **_predict_kwargs(x))[:2] for x in X_val]
        Y1hat_val = [t[0] for t in temp]
        Y2hat_val = [t[1] for t in temp]
        
        opt_prob_thresh1, opt_measure1, opt_nms_thresh1 = None, -np.inf, None
        opt_prob_thresh2, opt_measure2, opt_nms_thresh2 = None, -np.inf, None
        
        for _opt_nms_thresh in nms_threshs:
            _opt_prob_thresh1, _opt_measure1 = optimize_threshold(Y1_val, Y1hat_val, model=self, nms_thresh=_opt_nms_thresh, iou_threshs=iou_threshs, **optimize_kwargs)
            _opt_prob_thresh2, _opt_measure2 = optimize_threshold(Y2_val, Y2hat_val, model=self, nms_thresh=_opt_nms_thresh, iou_threshs=iou_threshs, **optimize_kwargs)
            
            if _opt_measure1 > opt_measure1:
                opt_prob_thresh1, opt_measure1, opt_nms_thresh1 = _opt_prob_thresh1, _opt_measure1, _opt_nms_thresh

            if _opt_measure2 > opt_measure2:
                opt_prob_thresh2, opt_measure2, opt_nms_thresh2 = _opt_prob_thresh2, _opt_measure2, _opt_nms_thresh

        opt_threshs1 = dict(prob=opt_prob_thresh1, nms=opt_nms_thresh1)
        opt_threshs2 = dict(prob=opt_prob_thresh2, nms=opt_nms_thresh2)

        self.thresholds1 = opt_threshs1
        self.thresholds2 = opt_threshs2
        print(end='', file=sys.stderr, flush=True)
        print("Using optimized values: prob_thresh1={prob:g}, nms_thresh1={nms:g}.".format(prob=self.thresholds1['prob'], nms=self.thresholds1['nms']))
        print("Using optimized values: prob_thresh2={prob:g}, nms_thresh2={nms:g}.".format(prob=self.thresholds2['prob'], nms=self.thresholds2['nms']))
        if save_to_json and self.basedir is not None:
            print("Saving to 'thresholds1.json' and 'thresholds2.json'.")
            save_json(opt_threshs1, str(self.logdir / 'thresholds1.json'))
            save_json(opt_threshs2, str(self.logdir / 'thresholds2.json'))
       
        return opt_threshs1, opt_threshs2


    def _guess_n_tiles(self, img):
        axes = self._normalize_axes(img, axes=None)
        shape = list(img.shape)
        if 'C' in axes:
            del shape[axes_dict(axes)['C']]
        b = self.config.train_batch_size**(1.0/self.config.n_dim)
        n_tiles = [int(np.ceil(s/(p*b))) for s,p in zip(shape,self.config.train_patch_size)]
        if 'C' in axes:
            n_tiles.insert(axes_dict(axes)['C'],1)
        return tuple(n_tiles)


    def _normalize_axes(self, img, axes):
        if axes is None:
            axes = self.config.axes
            assert 'C' in axes
            if img.ndim == len(axes)-1 and self.config.n_channel_in == 1:
                # img has no dedicated channel axis, but 'C' always part of config axes
                axes = axes.replace('C','')
        return axes_check_and_normalize(axes, img.ndim)


    def _compute_receptive_field(self, img_size=None):
        from scipy.ndimage import zoom
        if img_size is None:
            img_size = tuple(g*(128 if self.config.n_dim==2 else 64) for g in self.config.grid)
        if np.isscalar(img_size):
            img_size = (img_size,) * self.config.n_dim
        img_size = tuple(img_size)
        assert all(_is_power_of_2(s) for s in img_size)
        mid = tuple(s//2 for s in img_size)
        x = np.zeros((1,)+img_size+(self.config.n_channel_in,), dtype=np.float32)
        z = np.zeros_like(x)
        x[(0,)+mid+(slice(None),)] = 1

        output_dict  = self.keras_model.predict(x, verbose=0)
        
        y1 = output_dict['dist1'][0,...,0]
        y2 = output_dict['dist2'][0,...,0]

        output_dict0 = self.keras_model.predict(z, verbose=0)
        
        y10 = output_dict0['dist1'][0,...,0]
        y20 = output_dict0['dist2'][0,...,0]

        grid1 = tuple((np.array(x.shape[1:-1])/np.array(y1.shape)).astype(int))
        grid2 = tuple((np.array(x.shape[1:-1])/np.array(y2.shape)).astype(int))

        assert grid1 == self.config.grid
        assert grid2 == self.config.grid
        y1  = zoom(y1, grid1,order=0)
        y2  = zoom(y2, grid2,order=0)
        y10 = zoom(y10,grid1,order=0)
        y20 = zoom(y20,grid2,order=0)
        ind1 = np.where(np.abs(y1-y10)>0)
        ind2 = np.where(np.abs(y2-y20)>0)
        return [(m-np.min(i), np.max(i)-m) for (m,i) in zip(mid,ind1)], [(m-np.min(i), np.max(i)-m) for (m,i) in zip(mid,ind2)]

    def _axes_tile_overlap(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        try:
            self._tile_overlap1
            self._tile_overlap2
        except:
            self._tile_overlap1, self._tile_overlap2 = self._compute_receptive_field()
        overlap1 = dict(zip(
            self.config.axes.replace('C',''),
            tuple(max(rf) for rf in self._tile_overlap1)
        ))
        overlap2 = dict(zip(
            self.config.axes.replace('C',''),
            tuple(max(rf) for rf in self._tile_overlap2)
        ))
        return tuple(overlap1.get(a,0) for a in query_axes), tuple(overlap2.get(a,0) for a in query_axes)


    def export_TF(self, fname=None, single_output=True, upsample_grid=True):
        """Export model to TensorFlow's SavedModel format that can be used e.g. in the Fiji plugin

        Parameters
        ----------
        fname : str
            Path of the zip file to store the model
            If None, the default path "<modeldir>/TF_SavedModel.zip" is used
        single_output: bool
            If set, concatenates the two model outputs into a single output (note: this is currently mandatory for further use in Fiji)
        upsample_grid: bool
            If set, upsamples the output to the input shape (note: this is currently mandatory for further use in Fiji)
        """
        Concatenate, UpSampling2D, UpSampling3D, Conv2DTranspose, Conv3DTranspose = keras_import('layers', 'Concatenate', 'UpSampling2D', 'UpSampling3D', 'Conv2DTranspose', 'Conv3DTranspose')
        Model = keras_import('models', 'Model')

        if self.basedir is None and fname is None:
            raise ValueError("Need explicit 'fname', since model directory not available (basedir=None).")

        if self._is_multiclass():
            warnings.warn("multi-class mode not supported yet, removing classification output from exported model")

        grid = self.config.grid
        prob1 = self.keras_model.outputs[0][0]
        dist1 = self.keras_model.outputs[0][1]
        prob2 = self.keras_model.outputs[1][0]
        dist2 = self.keras_model.outputs[1][1]
        assert self.config.n_dim in (2,3)

        if upsample_grid and any(g>1 for g in grid):
            # CSBDeep Fiji plugin needs same size input/output
            # -> we need to upsample the outputs if grid > (1,1)
            # note: upsampling prob with a transposed convolution creates sparse
            #       prob output with less candidates than with standard upsampling
            conv_transpose = Conv2DTranspose if self.config.n_dim==2 else Conv3DTranspose
            upsampling     = UpSampling2D    if self.config.n_dim==2 else UpSampling3D
            prob1 = conv_transpose(1, (1,)*self.config.n_dim,
                                  strides=grid, padding='same',
                                  kernel_initializer='ones', use_bias=False)(prob1)
            dist1 = upsampling(grid)(dist1)
            prob2 = conv_transpose(1, (1,)*self.config.n_dim,
                                  strides=grid, padding='same',
                                  kernel_initializer='ones', use_bias=False)(prob2)
            dist2 = upsampling(grid)(dist2)

        inputs  = self.keras_model.inputs[0]
        outputs = Concatenate()([[prob1, dist1], [prob2, dist2]]) if single_output else [[prob1, dist1],[prob2, dist2]]
        csbdeep_model = Model(inputs, outputs)

        fname = (self.logdir / 'TF_SavedModel.zip') if fname is None else Path(fname)
        export_SavedModel(csbdeep_model, str(fname))
        return csbdeep_model



class StarDistPadAndCropResizer(Resizer):

    def __init__(self, grid, mode='reflect', **kwargs):
        assert isinstance(grid, dict)
        self.mode = mode
        self.grid = grid
        self.kwargs = kwargs


    def before(self, x, axes, axes_div_by):
        assert all(a%g==0 for g,a in zip((self.grid.get(a,1) for a in axes), axes_div_by))
        axes = axes_check_and_normalize(axes,x.ndim)
        def _split(v):
            return 0, v # only pad at the end
        self.pad = {
            a : _split((div_n-s%div_n)%div_n)
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        }
        x_pad = np.pad(x, tuple(self.pad[a] for a in axes), mode=self.mode, **self.kwargs)
        self.padded_shape = dict(zip(axes,x_pad.shape))
        if 'C' in self.padded_shape: del self.padded_shape['C']
        return x_pad


    def after(self, x, axes):
        # axes can include 'C', which may not have been present in before()
        axes = axes_check_and_normalize(axes,x.ndim)
        assert all(s_pad == s * g for s,s_pad,g in zip(x.shape,
                                                       (self.padded_shape.get(a,_s) for a,_s in zip(axes,x.shape)),
                                                       (self.grid.get(a,1) for a in axes)))
        crop = tuple (
            slice(0, -(math.floor(p[1]/g)) if p[1]>=g else None)
            for p,g in zip((self.pad.get(a,(0,0)) for a in axes),(self.grid.get(a,1) for a in axes))
        )
        return x[crop]


    def filter_points(self, ndim, points, axes):
        """ returns indices of points inside crop region """
        assert points.ndim==2
        axes = axes_check_and_normalize(axes,ndim)

        bounds = np.array(tuple(self.padded_shape[a]-self.pad[a][1] for a in axes if a.lower() in ('z','y','x')))
        idx = np.where(np.all(points< bounds, 1))
        return idx



def _tf_version_at_least(version_string="1.0.0"):
    from packaging import version
    return version.parse(tf.__version__) >= version.parse(version_string)
