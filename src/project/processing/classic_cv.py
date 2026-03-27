import cv2 as cv
from cv2.typing import MatLike, TermCriteria

from project.processing.pipeline import ImagePipeline, ImageState


def edge_method(
    process_pipeline: ImagePipeline,
    blur_ksize: tuple[int, int] = (5, 5),
    blur_stddev: float = 1.4,
    edge_th1: int = 60,
    edge_th2: int = 150,
    closing_ksize: tuple[int, int] = (3, 3),
    closing_iter=5,
    remove_object_size=1000,
) -> ImagePipeline:

    return (
        process_pipeline.gaussian_blur(blur_ksize, blur_stddev)
        .canny_edge_detect(edge_th1, edge_th2)
        .closing(closing_ksize, closing_iter)
        .remove_small_object(remove_object_size)
        .invert()
    )


def hsv_segmentation(
    pipeline: ImagePipeline,
    lower: MatLike,
    upper: MatLike,
    kernel_size: tuple[int, int],
    iters: int,
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
    processing_pipeline: ImagePipeline,
    k: int,
    criteria: TermCriteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        10,
        1.0,
    ),
) -> ImagePipeline:
    return processing_pipeline.k_means_clustering(k, criteria).invert()


def exessive_green_method(
    process_pipeline: ImagePipeline, threshold: int = 20, remove_object_size: int = 100
) -> ImagePipeline:
    return (
        process_pipeline.excessive_green_mask(threshold)
        .remove_small_object(remove_object_size)
        .invert()
    )
