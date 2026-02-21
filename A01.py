import numpy as np
import cv2
import gradio as gr
import matplotlib.pyplot as plt


def _finalize_lut(arr):
    arr = np.round(arr)
    arr = np.clip(arr, 0, 255)

    return arr.astype(np.uint8)

# Log Transform
def get_log_transform(max_r):
    r = np.arange(256)
    c = 255 / np.log(1 + max_r)
    s = c * np.log(1 + r)

    return _finalize_lut(s)

# Gamma Transform
def get_gamma_transform(gamma):
    r = np.arange(256)
    s = 255 * (r / 255) ** gamma

    return _finalize_lut(s)

# Histogram Equalization
def get_hist_equalize_transform(image, do_stretching):
    flat = image.flatten()
    hist = np.bincount(flat, minlength=256)
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)

    if do_stretching:
        cdf_min = np.min(cdf[cdf > 0])
        s = (cdf - cdf_min) / (1 - cdf_min) * 255
    else:
        s = cdf * 255

    return _finalize_lut(s)

# Piecewise linear
def get_piecewise_linear_transform(points):
    points = sorted(points, key=lambda x: x[0])

    r_vals, s_vals = zip(*points)
    r_vals = np.array(r_vals)
    s_vals = np.array(s_vals)

    r = np.arange(256)
    s = np.interp(r, r_vals, s_vals)

    return _finalize_lut(s)

def apply_intensity_transform(image, int_transform):
    return int_transform[image]

def plot_hist(image):
    hist = np.bincount(image.flatten(), minlength=256)
    fig = plt.figure()
    plt.bar(np.arange(256), hist)
    plt.title("Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    return fig

def plot_lut(lut):
    fig = plt.figure()
    plt.plot(lut)
    plt.title("Transformation Function")
    plt.xlabel("Input Intensity")
    plt.ylabel("Output Intensity")

    return fig

# Processing
def process_image(img, mode, gamma, max_r, stretch, points_text):
    if img is None:
        return None, None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mode == "Log":
        lut = get_log_transform(max_r)
    elif mode == "Gamma":
        lut = get_gamma_transform(gamma)
    elif mode == "Hist Equalize":
        lut = get_hist_equalize_transform(gray, stretch)
    elif mode == "Piecewise":
        try:
            pts = eval(points_text)
            pts = [tuple(p) for p in pts]
            lut = get_piecewise_linear_transform(pts)
        except:
            return None, None, None, None
    else:
        lut = np.arange(256, dtype=np.uint8)

    out = apply_intensity_transform(gray, lut)

    in_hist = plot_hist(gray)
    out_hist = plot_hist(out)
    lut_plot = plot_lut(lut)

    return out, in_hist, out_hist, lut_plot


####################################################################
# MAIN
####################################################################
def main():
    with gr.Blocks(title="CS470 - Assignment 01") as demo:
        gr.Markdown("## CS 470 – Intensity Transformations")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Image")

                mode = gr.Radio(
                    ["Log", "Gamma", "Hist Equalize", "Piecewise"],
                    value="Log",
                    label="Transformation"
                )
                gamma = gr.Slider(
                    0.1, 5.0, value=1.0, step=0.1,
                    label="Gamma"
                )
                max_r = gr.Slider(
                    1, 1000, value=255, step=1,
                    label="Max R (Log)"
                )
                stretch = gr.Checkbox(
                    label="Histogram Stretching"
                )
                points_text = gr.Textbox(
                    value="[[0,0],[255,255]]",
                    label="Piecewise Points [[r,s],...]"
                )
                btn = gr.Button("Apply")
            with gr.Column():
                output_img = gr.Image(label="Output Image")

                in_hist = gr.Plot(label="Input Histogram")
                out_hist = gr.Plot(label="Output Histogram")

                lut_plot = gr.Plot(label="Transformation Function")

        btn.click(
            process_image,
            inputs=[input_img, mode, gamma, max_r, stretch, points_text],
            outputs=[output_img, in_hist, out_hist, lut_plot]
        )
    demo.launch()


if __name__ == "__main__":
    main()