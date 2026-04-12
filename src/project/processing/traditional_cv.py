import cv2 as cv
from cv2.typing import MatLike, TermCriteria

from project.processing.pipeline import ImagePipeline, ImageState
from project.utils.constant import UPPER_GREEN, LOWER_GREEN


def edge_method(
    pipeline: ImagePipeline,
    blur_ksize: tuple[int, int] = (5, 5),
    blur_stddev: float = 1.4,
    edge_th1: int = 60,
    edge_th2: int = 150,
    closing_ksize: tuple[int, int] = (3, 3),
    closing_iter=5,
    remove_object_size=1000,
) -> ImagePipeline:

    return (
        pipeline.gaussian_blur(blur_ksize, blur_stddev)
        .canny_edge_detect(edge_th1, edge_th2)
        .closing(closing_ksize, closing_iter)
        .remove_small_object(remove_object_size)
        .invert()
    )


def hsv_segmentation_method(
    pipeline: ImagePipeline,
    lower: MatLike = LOWER_GREEN,
    upper: MatLike = UPPER_GREEN,
    kernel_size: tuple[int, int] = (3, 3),
    iters: int = 1,
) -> ImagePipeline:
    assert pipeline.image_state == ImageState.RGB
    return (
        pipeline.rgb_to_hsv()
        .in_color_range(lower, upper)
        .opening(kernel_size, iters=iters)
        .closing(kernel_size, iters=iters)
        .invert()
    )


def kmeans_method(
    pipeline: ImagePipeline,
    k: int,
    criteria: TermCriteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        10,
        1.0,
    ),
) -> ImagePipeline:
    return pipeline.k_means_clustering(k, criteria).invert()


def excessive_green_method(
    pipeline: ImagePipeline, threshold: int = 20, remove_object_size: int = 100
) -> ImagePipeline:
    return (
        pipeline.excessive_green_mask(threshold)
        .remove_small_object(remove_object_size)
        .invert()
    )


def watershed_method(
    pipeline: ImagePipeline, exg_low: int = -20, exg_high: int = 20
) -> ImagePipeline:
    return pipeline.watershed(exg_low=exg_low, exg_high=exg_high).invert()


def grabcut_method(pipeline: ImagePipeline) -> ImagePipeline:
    return (
        pipeline.grabcut(exg_threshold=10, iters=2)
        .opening((5, 5), iters=2)
        .remove_small_object(min_area=500)
        .invert()
    )


def crf_method(
    pipeline: ImagePipeline,
    exg_threshold: int = 10,
    gt_prob: float = 0.7,
    iters: int = 5,
    sxy_gaussian: int = 3,
    compat_gaussian: int = 3,
    sxy_bilateral: int = 60,
    srgb_bilateral: int = 13,
    compat_bilateral: int = 10,
    remove_object_size: int = 100,
) -> ImagePipeline:
    return (
        pipeline.dense_crf(
            exg_threshold=exg_threshold,
            gt_prob=gt_prob,
            iters=iters,
            sxy_gaussian=sxy_gaussian,
            compat_gaussian=compat_gaussian,
            sxy_bilateral=sxy_bilateral,
            srgb_bilateral=srgb_bilateral,
            compat_bilateral=compat_bilateral,
        )
        .remove_small_object(remove_object_size)
        .invert()
    )
