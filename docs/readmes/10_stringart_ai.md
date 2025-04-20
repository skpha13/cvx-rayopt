# StringArt-AI

StringArt-AI is a creative AI project that explores how models can learn to generate string art patterns from images. The core idea is simple: turn any input image into a string art representation, and train AI models to learn this transformation.

## Motivation

Given the fact that I can now generate a String Art representation for any input image, I can now build a dataset that pairs input images with their corresponding string art outputs. This enables training of machine learning models to predict string art configurations from new images.

## Dataset

I used the [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch) dataset as my input image source. The target outputs, string art representations, were generated procedurally using this StringArt Python package.

## Experiments

These input-output pairs were then used to train various AI models, aiming to replicate or predict the string art configurations from sketch-like inputs. The goal is for the model to learn a visual-to-pattern mapping that mimics the handcrafted style of traditional string art.

## Future Ideas

One observation during this project was that string art results tend to look better when the input images are sketch-like. This opens up a direction for improvement: applying a sketch transformation as a preprocessing step before generating the string art.

A great starting point for this approach is the following tutorial, which demonstrates how to convert images to pencil sketches using OpenCV:

[Image to Pencil Sketch using OpenCV](https://medium.com/@Kavya2099/image-to-pencil-sketch-using-opencv-ec3568443c5e)

## Learn More

You can find more details, code, and examples in the original repository: [StringArt-AI on GitHub](https://github.com/skpha13/StringArt-AI)
