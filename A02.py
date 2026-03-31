import cv2
import numpy as np
import gradio as gr

def read_kernel_file(filepath):
    with open(filepath, "r") as f:
        line = f.readline()
    tokens = line.split(" ")
    rowCnt = int(tokens[0])
    colCnt = int(tokens[1])
    values = np.array(tokens[2:], dtype=np.str_).astype(np.float64)
    kernel = values.reshape(rowCnt, colCnt)
    return kernel

def do_convolution_slow(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    flipped_kernel = cv2.flip(kernel, -1)
    img_h, img_w = image.shape
    k_h, k_w = flipped_kernel.shape

    pad_top = k_h // 2
    pad_bottom = k_h // 2
    pad_left = k_w // 2
    pad_right = k_w // 2
    padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    output = np.zeros((img_h, img_w), dtype=np.float64)
    for i in range(img_h):
        for j in range(img_w):
            total = 0.0
            for ki in range(k_h):
                for kj in range(k_w):
                    total = total + padded[i + ki, j + kj] * flipped_kernel[ki, kj]
            output[i, j] = total
    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
    return output

def do_convolution_fast(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    flipped_kernel = cv2.flip(kernel, -1)
    img_h, img_w = image.shape
    k_h, k_w = flipped_kernel.shape

    pad_top = k_h // 2
    pad_bottom = k_h // 2
    pad_left = k_w // 2
    pad_right = k_w // 2
    padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )
    output = np.zeros((img_h, img_w), dtype=np.float64)

    ki_indices = np.arange(k_h)
    kj_indices = np.arange(k_w)
    ki_grid, kj_grid = np.meshgrid(ki_indices, kj_indices, indexing='ij')
    ki_flat = ki_grid.ravel()
    kj_flat = kj_grid.ravel()
    kernel_flat = flipped_kernel.ravel()

    strides = padded.strides
    patches = np.lib.stride_tricks.as_strided(
        padded,
        shape=(img_h, img_w, k_h, k_w),
        strides=(strides[0], strides[1], strides[0], strides[1])
    )
    output = np.sum(patches * flipped_kernel[np.newaxis, np.newaxis, :, :], axis=(2, 3))

    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
    return output

def do_convolution_fourier(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    dft_h = cv2.getOptimalDFTSize(img_h + k_h - 1)
    dft_w = cv2.getOptimalDFTSize(img_w + k_w - 1)

    padded_image = np.zeros((dft_h, dft_w), dtype=np.float64)
    padded_image[:img_h, :img_w] = image
    padded_kernel = np.zeros((dft_h, dft_w), dtype=np.float64)
    padded_kernel[:k_h, :k_w] = kernel

    image_dft = np.fft.fft2(padded_image)
    kernel_dft = np.fft.fft2(padded_kernel)
    result_dft = image_dft * kernel_dft
    result = np.fft.ifft2(result_dft).real

    pad_top = k_h // 2
    pad_left = k_w // 2
    output = result[pad_top:pad_top + img_h, pad_left:pad_left + img_w]

    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
    return output

def apply_filter(input_image, kernel_file, alpha, beta):
    if input_image is None:
        return None
    if kernel_file is None:
        return None
    kernel = read_kernel_file(kernel_file)
    output = do_convolution_fast(
        image=input_image,
        kernel=kernel,
        alpha=alpha,
        beta=beta,
        convert_uint8=True,
    )
    return output

def apply_filter_fourier(input_image, kernel_file, alpha, beta):
    if input_image is None:
        return None
    if kernel_file is None:
        return None
    kernel = read_kernel_file(kernel_file)
    output = do_convolution_fourier(
        image=input_image,
        kernel=kernel,
        alpha=alpha,
        beta=beta,
        convert_uint8=True,
    )
    return output

def main():
    with gr.Blocks(title="CS 470 - A02 Convolution") as demo:
        gr.Markdown("# CS 470 - Assignment 02: Convolution")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image (Grayscale)",
                    type="numpy",
                    image_mode="L",
                )
                kernel_file = gr.File(
                    label="Kernel File (.txt)",
                    file_types=[".txt"],
                    type="filepath",
                )
                alpha = gr.Number(label="Alpha", value=1.0)
                beta = gr.Number(label="Beta", value=0.0)
                with gr.Row():
                    btn_fast = gr.Button("Apply Fast Convolution")
                    btn_fourier = gr.Button("Apply Fourier Convolution")
            with gr.Column():
                output_image = gr.Image(
                    label="Filtered Image",
                    type="numpy",
                    image_mode="L",
                )
        btn_fast.click(
            fn=apply_filter,
            inputs=[input_image, kernel_file, alpha, beta],
            outputs=output_image,
        )
        btn_fourier.click(
            fn=apply_filter_fourier,
            inputs=[input_image, kernel_file, alpha, beta],
            outputs=output_image,
        )
    demo.launch()
if __name__ == "__main__":
    main()