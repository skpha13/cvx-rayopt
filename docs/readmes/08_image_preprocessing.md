# Image Preprocessing

Some images, like the Lena input image, often have a dull or muted tone, resulting in a grayscale appearance that hinders accurate line detection by algorithms. 

Those methods shouldn't depend on some magic variable and should have the same tonality for every image.

To address this, I explored two preprocessing methods that aim to standardize the tone of images. These methods are consistent, meaning they enhance contrast for bland images while reducing it for overly dark ones, placing every image at a similar tonal level.

## Methods

### Histogram Equalization

This method works by distributing the histogram of an image evenly across the entire spectrum, effectively 'equalizing' it.

Algorithm steps:
    
1. Calculate the histogram `H` for the source image.
2. Normalize the histogram so that the sum of histogram bins is 255.
3. Compute the cumulative sum to get `H'`.
4. Transform the image using `H'` as a look-up table: `dst(x, y) = H'(src(x, y))`

For more details, check the official [OpenCV tutorial](https://docs.opencv.org/4.x/d4/d1b/tutorial_histogram_equalization.html).

<img src="../../imgs/lena_bw.png" alt="Lena input image" width=256>
<img src="../../imgs/airplane_bw.png" alt="Airplane input image" width=256>
<img src="../../imgs/tank.png" alt="Tank input image" width=256>
<img src="../../imgs/teddybear.png" alt="Teddybear input image" width=256>
<img src="../../imgs/butterfly_bw.png" alt="Butterfly input image" width=256>

> Input Images

<img src="../../outputs/preprocess/lena/histogram_equalization.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/airplane/histogram_equalization.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/tank/histogram_equalization.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/teddybear/histogram_equalization.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/butterfly/histogram_equalization.png" alt="Lena Histogram Equalization Preprocess" width=256>

> Input Images after Histogram Equalization

### Grayscale Quantization

Grayscale Quantization reduces the number of intensity levels in an image, effectively converting a continuous grayscale spectrum into a discrete, thresholded one.

Algorithm steps:

```math
Q(x) = \lfloor \frac{x}{\Delta} \rfloor * \Delta + \frac{\Delta}{2}
```

```math
\Delta = \frac{255}{L}
```

```math
L = \text{Desired Intensity Levels}
```

For more information, visit the [Wikipedia page](https://en.wikipedia.org/wiki/Quantization_(image_processing)#Grayscale_quantization).

<img src="../../imgs/lena_bw.png" alt="Lena input image" width=256>
<img src="../../imgs/airplane_bw.png" alt="Airplane input image" width=256>
<img src="../../imgs/tank.png" alt="Tank input image" width=256>
<img src="../../imgs/teddybear.png" alt="Teddybear input image" width=256>
<img src="../../imgs/butterfly_bw.png" alt="Butterfly input image" width=256>

> Input Images

<img src="../../outputs/preprocess/lena/grayscale_quantization_2_levels.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/airplane/grayscale_quantization_2_levels.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/tank/grayscale_quantization_2_levels.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/teddybear/grayscale_quantization_2_levels.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/butterfly/grayscale_quantization_2_levels.png" alt="Lena Histogram Equalization Preprocess" width=256>

> Input Images after Grayscale Quantization

### Sketch Effect

This preprocessing method simulates a **pencil sketch** from a grayscale image using a series of image transformations.

Algorithm steps:
    
1. Invert the input image.
2. Apply Gaussian blur to the inverted image.
3. Invert the blurred result.
4. Perform pixel-wise division between the original image and the inverted blurred image (this mimics a "color dodge" blend).
5. Apply histogram equalization to enhance contrast in the final output.

For more details, check this [Medium Article](https://medium.com/@Kavya2099/image-to-pencil-sketch-using-opencv-ec3568443c5e).

<img src="../../imgs/lena_bw.png" alt="Lena input image" width=256>
<img src="../../imgs/airplane_bw.png" alt="Airplane input image" width=256>
<img src="../../imgs/tank.png" alt="Tank input image" width=256>
<img src="../../imgs/teddybear.png" alt="Teddybear input image" width=256>
<img src="../../imgs/butterfly_bw.png" alt="Butterfly input image" width=256>

> Input Images

<img src="../../outputs/preprocess/lena/sketch_effect.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/airplane/sketch_effect.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/tank/sketch_effect.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/teddybear/sketch_effect.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/butterfly/sketch_effect.png" alt="Lena Histogram Equalization Preprocess" width=256>

> Input Images after Sketch Effect

## Side by Side Comparison

### Lena

<img src="../../imgs/lena_bw.png" alt="Lena input image" width=256>
<img src="../../outputs/preprocess/lena/histogram_equalization.png" alt="Lena Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/lena/grayscale_quantization_2_levels.png" alt="Lena Grayscale Quantization 2 Levels Preprocess" width=256>
<img src="../../outputs/preprocess/lena/sketch_effect.png" alt="Lena Sketch Effect" width=256>

### Airplane

<img src="../../imgs/airplane_bw.png" alt="Airplane input image" width=256>
<img src="../../outputs/preprocess/airplane/histogram_equalization.png" alt="Airplane Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/airplane/grayscale_quantization_2_levels.png" alt="Airplane Grayscale Quantization 2 Levels Preprocess" width=256>
<img src="../../outputs/preprocess/airplane/sketch_effect.png" alt="Airplane Sketch Effect" width=256>

### Butterfly

<img src="../../imgs/butterfly_bw.png" alt="Butterfly input image" width=256>
<img src="../../outputs/preprocess/butterfly/histogram_equalization.png" alt="Butterfly Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/butterfly/grayscale_quantization_2_levels.png" alt="Butterfly Grayscale Quantization 2 Levels Preprocess" width=256>
<img src="../../outputs/preprocess/butterfly/sketch_effect.png" alt="Butterfly Sketch Effect" width=256>

### Tank

<img src="../../imgs/tank.png" alt="Tank input image" width=256>
<img src="../../outputs/preprocess/tank/histogram_equalization.png" alt="Tank Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/tank/grayscale_quantization_2_levels.png" alt="Tank Grayscale Quantization 2 Levels Preprocess" width=256>
<img src="../../outputs/preprocess/tank/sketch_effect.png" alt="Tank Sketch Effect" width=256>

### Teddybear

<img src="../../imgs/teddybear.png" alt="Teddybear input image" width=256>
<img src="../../outputs/preprocess/teddybear/histogram_equalization.png" alt="Teddybear Histogram Equalization Preprocess" width=256>
<img src="../../outputs/preprocess/teddybear/grayscale_quantization_2_levels.png" alt="Teddybear Grayscale Quantization 2 Levels Preprocess" width=256>
<img src="../../outputs/preprocess/teddybear/sketch_effect.png" alt="Teddybear Sketch Effect" width=256>

## Analysis

### Result Impact

<img src="../../outputs/preprocess/lena/preprocessed_results.png" alt="Image Processing Stages">
<img src="../../outputs/preprocess/airplane/preprocessed_results.png" alt="Image Processing Stages">
<img src="../../outputs/preprocess/butterfly/preprocessed_results.png" alt="Image Processing Stages">
<img src="../../outputs/preprocess/tank/preprocessed_results.png" alt="Image Processing Stages">
<img src="../../outputs/preprocess/teddybear/preprocessed_results.png" alt="Image Processing Stages">

**Grayscale Quantization**, being more aggressive, simplifies shape detection due to the binary nature of its output. In contrast, **Histogram Equalization** offers a subtler adjustment, retaining more image details.

The **Sketch Effect** introduces enhanced edge details and produces visually rich outputs. As observed in the results for images like lena, airplane, butterfly, and tank, the sketch effect highlights contours effectively.

However, this enhancement can also introduce **drawbacks**. In the airplane and tank images, for instance, the primary subject sometimes blends with the background due to increased noise and amplified gradients, thus making it harder to isolate the main object.

### Difference Analysis

<img src="../../outputs/preprocess/lena/diff_analysis.png" alt="Difference Analysis">
<img src="../../outputs/preprocess/airplane/diff_analysis.png" alt="Difference Analysis">
<img src="../../outputs/preprocess/butterfly/diff_analysis.png" alt="Difference Analysis">
<img src="../../outputs/preprocess/tank/diff_analysis.png" alt="Difference Analysis">
<img src="../../outputs/preprocess/teddybear/diff_analysis.png" alt="Difference Analysis">

**Grayscale Quantization** shows a more pronounced difference compared to the raw output, especially around the outer face areas, while **Histogram Equalization** maintains finer details.

The most aggressive transformation is introduced by the **Sketch Effect**. It significantly alters the image by amplifying both edges and noise. While it promotes finer detail visibility, it also increases background interference, making it both the most visually striking and the most distortion-prone among the preprocessing methods.

### Overlay Analysis

<img src="../../outputs/preprocess/lena/overlay_analysis.png" alt="Overlay Analysis">
<img src="../../outputs/preprocess/airplane/overlay_analysis.png" alt="Overlay Analysis">
<img src="../../outputs/preprocess/butterfly/overlay_analysis.png" alt="Overlay Analysis">
<img src="../../outputs/preprocess/tank/overlay_analysis.png" alt="Overlay Analysis">
<img src="../../outputs/preprocess/teddybear/overlay_analysis.png" alt="Overlay Analysis">

The raw output is marked in red, while the preprocessed results appear in blue. Areas where both overlap are shown in purple.

A notable observation is that the **Sketch Effect** produces significantly more unique blue lines, especially in background areas. This suggests that the sketch preprocessing introduces additional details and noise that were not present in the raw solution, highlighting its more aggressive transformation.

### Fast Fourier Transform Analysis

<img src="../../outputs/preprocess/lena/fft_analysis.png" alt="FFT Analysis">
<img src="../../outputs/preprocess/airplane/fft_analysis.png" alt="FFT Analysis">
<img src="../../outputs/preprocess/butterfly/fft_analysis.png" alt="FFT Analysis">
<img src="../../outputs/preprocess/tank/fft_analysis.png" alt="FFT Analysis">
<img src="../../outputs/preprocess/teddybear/fft_analysis.png" alt="FFT Analysis">

Despite the preprocessing differences, the frequency spectrum after FFT remains consistent across all methods, indicating that tone adjustments do not affect the frequency content.
