#from django.db import models
from django.db import models as djmodels
from django.shortcuts import render
from django.conf import settings
from django import forms

from modelcluster.fields import ParentalKey

from wagtail.admin.edit_handlers import (
    FieldPanel,
    MultiFieldPanel,
    InlinePanel,
    StreamFieldPanel,
    PageChooserPanel,
)
from wagtail.core.models import Page, Orderable
from wagtail.core.fields import RichTextField, StreamField
from wagtail.images.edit_handlers import ImageChooserPanel
from django.core.files.storage import default_storage

from pathlib import Path

from streams import blocks

import sqlite3, datetime, os, uuid, glob

### Import for DeOldify project
import fastai
from deoldify.visualize import *
import warnings
from urllib.parse import urlparse
# import os
###

### Import for Style Transfer project
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
            ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

## check if Eager execution is enabled
print("Eager execution: {}".format(tf.executing_eagerly()))
## END - Import for Style Transfer project

str_uuid = uuid.uuid4()  # The UUID for image uploading

## Style Transfer functions
def load_img(path_to_img):
  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  return img

def imshow(img, title=None):
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display 
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)

def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  # resize image to match with customized trained model
  img = tf.image.resize(img, (224, 224))
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x

def get_model():
    """ Creates our model with access to intermediate layers. 
  
    This function will load the VGG19 model and access the intermediate layers. 
    These layers will then be used to create a new model that will take input image
    and return the outputs from these intermediate layers from the VGG model. 
  
    Returns:
        returns a keras model that takes image inputs and outputs the style and 
        content intermediate layers. 
    """
    content_layers = ['block5_conv2'] 

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1'
                ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    # Load our model. We load pretrained VGG, trained on imagenet data
    # vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg = tf.keras.models.load_model('dummy/models/20220529_09_cnn1.h5')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers 
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model 
    return models.Model(vgg.input, model_outputs)

def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
  # We make the image channels first 
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
  """Expects two images of dimension h, w, c"""
  # height, width, num filters of each layer
  # We scale the loss at a given layer by the size of the feature map and the number of filters
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)

def get_feature_representations(model, content_path, style_path):
  """Helper function to compute our content and style feature representations.

  This function will simply load and preprocess both the content and style 
  images from their path. Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    content_path: The path to the content image.
    style_path: The path to the style image
    
  Returns:
    returns the style features and the content features. 
  """
  # Load our images in 
  content_image = load_and_process_img(content_path)
  style_image = load_and_process_img(style_path)
  
  # batch compute content and style features
  style_outputs = model(style_image)
  content_outputs = model(content_image)
  
  
  # Get the style and content feature representations from our model  
  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
  """This function will compute the loss total loss.
  
  Arguments:
    model: The model that will give us access to the intermediate layers
    loss_weights: The weights of each contribution of each loss function. 
      (style weight, content weight, and total variation weight)
    init_image: Our initial base image. This image is what we are updating with 
      our optimization process. We apply the gradients wrt the loss we are 
      calculating to this image.
    gram_style_features: Precomputed gram matrices corresponding to the 
      defined style layers of interest.
    content_features: Precomputed outputs from defined content layers of 
      interest.
      
  Returns:
    returns the total loss, style loss, content loss, and total variational loss
  """
  style_weight, content_weight = loss_weights
  
  # Feed our init image through our model. This will give us the content and 
  # style representations at our desired layers. Since we're using eager
  # our model is callable just like any other function!
  model_outputs = model(init_image)
  
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
  
  style_score = 0
  content_score = 0

  # Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
  # Accumulate content losses from all layers 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
  style_score *= style_weight
  content_score *= content_weight

  # Get total loss
  loss = style_score + content_score 
  return loss, style_score, content_score

def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss

# import IPython.display

def run_style_transfer(content_path
                        ,style_path
                        ,num_iterations=100
                        ,content_weight=1e3 
                        ,style_weight=1e-2): 
    # We don't need to (or want to) train any layers of our model, so we set their
    # trainable to false. 

    content_layers = ['block5_conv2'] 

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1'
                ]

    model = get_model() 
    for layer in model.layers:
        layer.trainable = False
  
    # Get the style and content feature representations (from our specified intermediate layers) 
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    # for tensorflow 1.X
    # opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # For displaying intermediate images 
    iter_count = 1
  
    # Store our best result
    best_loss, best_img = float('inf'), None
  
    # Create a nice config 
    loss_weights = (style_weight, content_weight)
    cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
    }
    
    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()
  
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   
  
    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time() 
    
    if loss < best_loss:
        # Update best loss and best image from total loss. 
        best_loss = loss
        best_img = deprocess_img(init_image.numpy())

    if i % display_interval== 0:
        start_time = time.time()
      
        # Use the .numpy() method to get the concrete numpy array
        plot_img = init_image.numpy()
        plot_img = deprocess_img(plot_img)
        imgs.append(plot_img)
        # IPython.display.clear_output(wait=True)
        # IPython.display.display_png(Image.fromarray(plot_img))
        print('Iteration: {}'.format(i))        
        print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
        print('Total time: {:.4f}s'.format(time.time() - global_start))
        # IPython.display.clear_output(wait=True)
        # plt.figure(figsize=(14,4))
        # for i,img in enumerate(imgs):
            # plt.subplot(num_rows,num_cols,i+1)
            # plt.imshow(img)
            # plt.xticks([])
            # plt.yticks([])
      
    return best_img, best_loss

## End - Style Transfer functions


def reset():
    print(f'Entering reset()...')
    files_result = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/Result/*.*')), recursive=True)
    files_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/image/*.*')), recursive=True)
    style_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/style/*.*')), recursive=True)
    files = []
    if len(files_result) != 0:
        files.extend(files_result)
    if len(files_upload) != 0:
        files.extend(files_upload)
    if len(style_upload) != 0:
        files.extend(style_upload)
    if len(files) != 0:
        for f in files:
            try:
                if (not (f.endswith(".txt"))):
                    os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        file_li = [Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'),
                   Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'),
                   Path(f'{settings.MEDIA_ROOT}/uploadedPics/style_list.txt'),
                   Path(f'{settings.MEDIA_ROOT}/Result/stats.txt')]
        for p in file_li:
            file = open(Path(p), "r+")
            file.truncate(0)
            file.close()

# Create your models here.
class ImagePage(Page):
    """Image Page."""

    template = "StyleTransfer\StyleTransfer.html"

    max_count = 2

    name_title = djmodels.CharField(max_length=100, blank=True, null=True)
    name_subtitle = RichTextField(features=["bold", "italic"], blank=True)

    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("name_title"),
                FieldPanel("name_subtitle"),

            ],
            heading="Page Options",
        ),
    ]


    def reset_context(self, request):
        context = super().get_context(request)
        context["my_uploaded_file_names"]= []
        context["my_uploaded_style_img"]= []
        context["my_result_file_names"]=[]
        context["my_staticSet_names"]= []
        context["my_lines"]: []
        return context

    def serve(self, request):

        # Style transfer config
        content_layers = ['block5_conv2'] 

        # Style layer we are interested in
        style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1'
                ]

        content_path = 'N:\\Data_Work\\20220301 42028 Deep Learning and Convolutional Neural Network\\Assignment 3\\Resources\\Neural Style Transfer\\data\\GEDC0256.JPG'
        style_path = 'N:\\Data_Work\\20220301 42028 Deep Learning and Convolutional Neural Network\\Assignment 3\\Resources\\Neural Style Transfer\\data\\The_Great_Wave_off_Kanagawa.jpg'
        
        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)
        # End - Style transfer config

        context = self.reset_context(request)
        # reset()
        emptyButtonFlag = False
        if request.POST.get('start')=="":
            print(request.POST.get('start'))
            print("Start own code here with Style Transfer")
            # put your detection code here
            warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
            file_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/image/*.jpg')), recursive=True)
            files = []
            files.extend(file_upload)

            style_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/style/*.jpg')), recursive=True)
            styles = []
            styles.extend(style_upload)

            ## Style transfer
            before_file = files[0]
            img_dim_save = load_img(before_file).shape
            # print(img_dim_save)
            # print(f'DimX: {img_dim_save[1]}')
            # print(f'DimY: {img_dim_save[2]}')

            after_image, best_loss = run_style_transfer(
                before_file   # content_path
                ,styles[0]
                ,num_iterations=50
                )

            after_image_c = tf.image.resize(after_image, (img_dim_save[1], img_dim_save[2]))
            after_image_cv2 = cv2.cvtColor(np.array(after_image_c),cv2.COLOR_RGB2BGR)
            cv2.imwrite('media/Result/colorized-local.jpg', after_image_cv2)
            context["my_result_file_names"].append('\media\Result\colorized-local.jpg')
            # print(context["my_result_file_names"])

            ## Display uploaded content and style images
            context["my_uploaded_file_names"] = files
            context["my_uploaded_style_img"]= styles

            return render(request, "StyleTransfer\StyleTransfer.html", context)
            ## END - Style transfer

        if request.POST.get('startDeoldify')=="":
            print(request.POST.get('start'))
            print("Start own code here with Deoldify")
            # put your detection code here
            warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
            file_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/image/*.jpg')), recursive=True)
            files = []
            files.extend(file_upload)

            # style_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/style/*.jpg')), recursive=True)
            # styles = []
            # styles.extend(style_upload)

            ## Style transfer
            before_file = files[0]

            RENDER_FACTOR = 35 # 35  
            WATERMARK = False

            colorizer = get_image_colorizer(root_folder=Path(f'{settings.MEDIA_ROOT}/uploadedPics'),artistic=True)

            after_image = colorizer.get_transformed_image(
                before_file
                ,render_factor=RENDER_FACTOR
                ,watermarked=WATERMARK
                )

            after_image_cv2 = cv2.cvtColor(np.array(after_image),cv2.COLOR_RGB2BGR)
            cv2.imwrite('media/Result/colorized-local.jpg', after_image_cv2)
            context["my_result_file_names"].append('\media\Result\colorized-local.jpg')
            print(context["my_result_file_names"])
            # return render(request, "cam_app2\image.html", context)
            return render(request, "StyleTransfer\StyleTransfer.html", context)
            # return context
            ## END - Deoldify

        ## Display uploaded images
        if (request.FILES and emptyButtonFlag == False):
            print("reached here files")
            reset()
            self.reset_context(request)
            context["my_uploaded_file_names"] = []
            for file_obj in request.FILES.getlist("file_image"):
                uuidStr = uuid.uuid4()
                filename = f"{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}"
                with default_storage.open(Path(f"uploadedPics/image/{filename}"), 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)
                filename = Path(f"{settings.MEDIA_URL}uploadedPics/image/{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}")
                with open(Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'), 'a') as f:
                    f.write(str(filename))
                    f.write("\n")

                context["my_uploaded_file_names"].append(str(f'{str(filename)}'))

            context["my_uploaded_style_img"]= []
            for file_obj in request.FILES.getlist("file_style"):
                uuidStr = uuid.uuid4()
                filename = f"{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}"
                with default_storage.open(Path(f"uploadedPics/style/{filename}"), 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)
                filename = Path(f"{settings.MEDIA_URL}uploadedPics/style/{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}")
                with open(Path(f'{settings.MEDIA_ROOT}/uploadedPics/style_list.txt'), 'a') as f:
                    f.write(str(filename))
                    f.write("\n")

                context["my_uploaded_style_img"].append(str(f'{str(filename)}'))
            return render(request, "StyleTransfer\StyleTransfer.html", context)

        return render(request, "StyleTransfer\StyleTransfer.html", {'page': self})
