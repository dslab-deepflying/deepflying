
# 用tf.keras进行神经风格迁移

# 概述
在本教程中，我们将学习如何使用深度学习来实现用另一张图像的风格来组合图像(甚至希望您能像毕加索或者梵高一样绘制?)这被称为神经风格转移（neural style transfer）!
这是 Leon A. Gatys的论文《A Neural Algorithm of Artistic Style》中概述的一项算法，非常适合阅读，你一定要看看它。
但是。什么是神经风格转移?
神经风格转移是一种优化技术，它用三张图像,分别是内容图像、样式风格参考图像(如一个著名画家的作品)和你想要被风格化处理的输入图像,混合在一起,这样输入图像内容会被转化，看起来很像内容图像，但用了风格图像的样式来“绘画”。
举个例子，左上角的图片是一副原图，我们采用不同名家画作的风格来进行“绘画”，得到的结果如下：
![%E5%9B%BE%E7%89%871.png](attachment:%E5%9B%BE%E7%89%871.png)
这是魔法还是深度学习?幸运的是，这并不涉及任何魔法:风格迁移是一个非常有趣的技术，它展示了神经网络的功能及内部表示。
神经风格传递的原理是定义两个距离函数。一个是![image.png](attachment:image.png)，它描述了两幅图像的内容有多么不同，另一个是![image.png](attachment:image.png)，它描述了两幅图像的风格差异。然后，给三幅图像，一个期望的风格图像、一个期望的内容图像和一张输入图像(用内容图像初始化)，我们尝试对输入图像进行转换，来最小化与内容图像的内容距离及其与样式图像的样式距离。总之，我们将获取基本输入图像、要匹配的内容图像和要匹配的样式图像。我们用反向传播算法最小化内容和样式距离(损失)来转换基本输入图像，创建一个匹配内容图像的内容和样式图像的样式的图像。

# 具体概念包括:
在这个过程中，我们将围绕以下概念建立实践经验、培养直觉
Eager Execution——使用TensorFlow的命令式编程环境来快速编程。
了解更多关于Eager Execution
了解它的参考文档
使用Functional API来定义模型——我们将构建模型的子集，它将允许我们使用Functional API访问必要的中间激活。
利用预训练模型的特征映射——学习如何使用预训练模型及其特征映射
创建自定义训练循环——我们将研究如何设置优化器，使给定的输入参数损失最小化


# 我们将按照一般步骤进行风格迁移:
1. 可视化数据
2. 基本数据预处理/准备
3. 设置损失函数
4. 创建模型
5. 优化损失函数
读者:这篇文章面向的是熟悉基本机器学习概念的中级用户。要充分利用这篇文章，你应该:
阅读Gaty的论文——我们将在此过程中进行解释，但论文将提供一个更全面的说明。
理解用梯度下降来降低损失值
预计时间:30分钟

# 准备

# 下载图片


```python
import os
img_dir = '/tmp/nst'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg
!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg
!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg
!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg
!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg
!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg
```

# 导入和配置模块


```python
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools
```


```python
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
```

我们将从启用eager execution开始。eager execution使我们能够以最清晰和最易读的方式完成这项技术。


```python
tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))
```


```python
# Set up some global values here
content_path = '/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg'
style_path = '/tmp/nst/The_Great_Wave_off_Kanagawa.jpg'
```

# 可视化输入


```python
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
```


```python
def imshow(img, title=None):
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display 
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)
```

这些是输入内容和样式图像。我们希望“创建”一个图像，它使用内容图像的内容，但是采用风格图像的风格。


```python
plt.figure(figsize=(10,10))

content = load_img(content_path).astype('uint8')
style = load_img(style_path).astype('uint8')

plt.subplot(1, 2, 1)
imshow(content, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
plt.show()
```

# 准备数据
让我们创建一些方法来轻松地加载和预处理图像。
根据VGG训练流程，我们执行与预期相同的预处理流程。VGG网络在图像上进行训练，每个信道通过均值= [103.939,116.779,123.68]和信道BGR进行归一化


```python

def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img
```


为了查看优化器的输出，我们需要执行逆预处理步骤。此外，由于优化后的图像的值可能介于$- \infty$和$\infty$之间，因此我们必须进行裁剪以将值保持在0-255范围内。


```python

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x
```

# 定义内容和样式表示
为了获得图像的内容和风格表示，我们将查看模型中的一些中间层。随着我们越来越深入到模型中，这些中间层表示越来越高阶的特征。在本例中，我们使用的是网络架构VGG19，这是一个预训练的图像分类网络。这些中间层对于定义我们图像内容和样式是非常必要的。对于输入图像，我们将在这些中间层上尝试匹配相应的样式和内容目标表示。
为什么需要中间层次?
您可能想知道，为什么我们预先训练的图像分类网络中的这些中间输出允许我们定义样式和内容表示。在高层次上，这一现象可以解释为：为了让网络执行图像分类(我们的网络已被训练过)，它必须理解图像。这包括将原始图像作为输入像素，并通过将原始图像像素转换为对图像中现有特性的复杂理解来构建内部表示。这也是卷积神经网络能够很好概括的部分原因:它们能够捕获不变性并定义与背景噪声和其他麻烦无关的类（例如，猫与狗）之间的特征。因此，在输入原始图像和输出分类标签之间的某个地方，模型充当一个复杂的特征提取器;因此，通过访问中间层，我们能够描述输入图像的内容和样式。
具体来说，我们将从我们的网络中提取这些中间层


```python
# Content layer where will pull our feature maps
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
```

# 构建模型
在这种情况下，我们加载VGG19，并将我们的输入张量输入到模型中。这将允许我们提取内容，样式和生成的图像的特征映射（以及随后的内容和样式表示）。
如本文所述，我们使用VGG19。此外，由于VGG19是一个相对简单的模型(与ResNet、Inception等相比)，特征映射实际上更适合于风格迁移。
为了访问与我们的风格和内容特征映射相对应的中间层，我们获得了相应的输出，并使用KerasFunctional API，使用所需的输出激活定义了我们的模型。
使用Functional API定义模型只需定义输入和输出:
model = Model(inputs, outputs)
模型=模型(输入，输出)


```python

def get_model():
  """ Creates our model with access to intermediate layers. 
  
  This function will load the VGG19 model and access the intermediate layers. 
  These layers will then be used to create a new model that will take input image
  and return the outputs from these intermediate layers from the VGG model. 
  
  Returns:
    returns a keras model that takes image inputs and outputs the style and 
      content intermediate layers. 
  """
  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  # Get output layers corresponding to style and content layers 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  # Build model 
  return models.Model(vgg.input, model_outputs)
```


在上面的代码片段中，我们将加载预训练的图像分类网络。然后我们获取前面定义的感兴趣的层。然后通过将模型的输入设置为图像，将输出设置为样式层和内容层的输出来定义模型。换句话说，我们创建了一个模型，它将接受一个输入图像并输出内容和样式的中间层!


# 定义并创建我们的损失函数(内容和样式距离)


# 内容损失
我们的内容损失定义实际上非常简单。我们将向网络传递所需的内容图像和基本输入图像。这将从我们的模型返回中间层输出(来自上面定义的层)。然后我们简单地取这些图像的两个中间表示之间的欧氏距离。
更正式地说，内容损失是一个函数，它描述内容与输出图像$x$和内容图像$p$之间的距离。设$C_{nn}$为预训练的深度卷积神经网络。同样，在本例中我们使用VGG19。设$X$为任意图像，则$C_{nn}(X)$是由X提供的网络。然后，我们将内容距离(损失)正式描述为:$$L^l_{content}(p, x) = \sum_{i, j}(F^l_{ij}(x) - p ^l_{ij}(p))^2$$
我们以通常的方式执行反向传播，以使内容损失最小化。因此，我们更改初始图像，直到它在某个层(在content_layer中定义)中生成与原始内容图像类似的响应。
这可以很简单地实现。同样，它将把网络L层的特征映射作为输入，由输入图像x和内容图像p提供，并返回内容距离。


# 计算内容损失
实际上，我们将在每个需要的层添加内容损失。这样，当我们通过模型输入图像时，每次迭代都会正确地计算通过模型的所有内容损失，所有的梯度都会计算出来。


```python
def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))
```

# 风格损失
计算风格损失涉及的更多，但遵循相同的原则，这一次向网络提供基本输入图像和样式风格图像。但是，我们没有比较基本输入图像和样式风格图像的原始中间输出，而是比较了这两个输出的Gram矩阵。
在数学上，我们将基本输入图像$x$和样式风格图像$a$的风格损失描述为这些图像的样式表示(gram矩阵)之间的距离。我们将图像的样式表示描述为Gram矩阵$G^l$给出的不同过滤响应之间的相关性，其中$G^ {ij}$是矢量化特征映射$i$和$j$ in层$l$之间的内积。我们可以看到，在给定图像的feature map上生成的$G^l_{ij}$表示feature map $i$和$j$之间的相关性。
为了生成基本输入图像的样式，我们从内容图像执行梯度下降，将其转换为与原始图像的样式表示相匹配的图像。我们通过最小化样式图像和输入图像的特征相关映射之间的平均平方距离来实现这一点。每个层对总样式损失的贡献由$$E_l = \frac{1}{4N_l^2M_l^2} \sum_{i,j}(G^l_{ij} - A^l_{ij})^2$$描述
其中$G^ {ij}$和$A^ {ij}$分别是$x$和$A $在$l$层中的样式表示形式。$N_l$描述特征映射的数量，每个特征映射的大小$M_l = height * width$。因此，每一层的总样式损失是$$L_{style}(a, x) = \sum_{l \in l} w_l E_l$$，其中我们将每一层的样式损失的贡献加权到某个因子$w_l$。在本例中，我们对每一层的权重相等($w_l =\frac{1}{|L|}$)


# 计算风格损失
同样，我们用距离来度量我们的风格损失。


```python
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
```

# 应用样式风格转移到我们的图像


# 运行梯度下降
如果你不熟悉梯度下降/反向传播或需要复习，你一定要看看这个很棒的资源。
在本例中，我们使用Adam*优化器来最小化损失。我们迭代地更新输出图像，以便最大限度的减少损失:我们不更新与网络相关的权重，而是训练输入图像，使损失最小化。为了做到这一点，我们必须知道如何计算损失和梯度。
L-BFGS *请注意,如果您熟悉此算法，建议不要使用L-BFGS，因为本教程背后的主要动机是用eager execution来说明最佳实践，并且通过使用Adam，我们可以演示用自定义训练循环来实现的自动编程/梯度带功能。
我们将定义一个小辅助函数，它将加载内容和样式风格图像，通过我们的网络向前传播它们，然后从我们的模型输出内容和样式特征表示。


```python
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
```

# 计算损失和梯度
这里我们用 tf.GradientTape来计算梯度。它允许我们利用自动区分，通过跟踪操作来计算后面的梯度。它记录了前向传播过程中的操作，然后能够计算后向传播过程中损失函数相对于输入图像的梯度。


```python
# def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
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
```


然后计算梯度很简单。


```python
def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss
```

# 优化循环Optimization Loop


```python

import IPython.display

def run_style_transfer(content_path, 
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3, 
                       style_weight=1e-2): 
  # We don't need to (or want to) train any layers of our model, so we set their
  # trainable to false. 
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False
  
  # Get the style and content feature representations (from our specified intermediate layers) 
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  # Set initial image
  init_image = load_and_process_img(content_path)
  init_image = tfe.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

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
      IPython.display.clear_output(wait=True)
      IPython.display.display_png(Image.fromarray(plot_img))
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
  print('Total time: {:.4f}s'.format(time.time() - global_start))
  IPython.display.clear_output(wait=True)
  plt.figure(figsize=(14,4))
  for i,img in enumerate(imgs):
      plt.subplot(num_rows,num_cols,i+1)
      plt.imshow(img)
      plt.xticks([])
      plt.yticks([])
      
  return best_img, best_loss
```


```python

best, best_loss = run_style_transfer(content_path, 
                                     style_path, num_iterations=1000)

```


```python
Image.fromarray(best)
```


若要从Colab下载图像，请取消以下代码的注释:


```python
#from google.colab import files
#files.download('wave_turtle.png')
```

# 可视化输出
我们对输出图像进行“后处理”，以删除应用于其上的处理
We "deprocess" the output image in order to remove the processing that was applied to it.


```python
def show_results(best_img, content_path, style_path, show_large_final=True):
  plt.figure(figsize=(10, 5))
  content = load_img(content_path) 
  style = load_img(style_path)

  plt.subplot(1, 2, 1)
  imshow(content, 'Content Image')

  plt.subplot(1, 2, 2)
  imshow(style, 'Style Image')

  if show_large_final: 
    plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()
```


```python
show_results(best, content_path, style_path)
```

# 在其他图像上尝试
Tuebingen的图像

摄影：Andreas Praefcke [GFDL（http://www.gnu.org/copyleft/fdl.html）或CC BY 3.0（https://creativecommons.org/licenses/by/3.0）]，来自Wikimedia Commons

# Starry night + Tuebingen


```python
best_starry_night, best_loss = run_style_transfer('/tmp/nst/Tuebingen_Neckarfront.jpg',
                                                  '/tmp/nst/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')
```


```python
show_results(best_starry_night, '/tmp/nst/Tuebingen_Neckarfront.jpg',
             '/tmp/nst/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')
```

# Pillars of Creation + Tuebingen


```python
best_poc_tubingen, best_loss = run_style_transfer('/tmp/nst/Tuebingen_Neckarfront.jpg', 
                                                  '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')
```


```python
show_results(best_poc_tubingen, 
             '/tmp/nst/Tuebingen_Neckarfront.jpg',
             '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')
```

# Kandinsky Composition 7 + Tuebingen


```python
best_kandinsky_tubingen, best_loss = run_style_transfer('/tmp/nst/Tuebingen_Neckarfront.jpg', 
                                                  '/tmp/nst/Vassily_Kandinsky,_1913_-_Composition_7.jpg')
```


```python

show_results(best_kandinsky_tubingen, 
             '/tmp/nst/Tuebingen_Neckarfront.jpg',
             '/tmp/nst/Vassily_Kandinsky,_1913_-_Composition_7.jpg')
```

# Pillars of Creation + Sea Turtle


```python
best_poc_turtle, best_loss = run_style_transfer('/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg', 
                                                  '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')
```


```python

show_results(best_poc_turtle, 
             '/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg',
             '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')
```

# 关键点

# 包括:
我们建立了几个不同的损失函数，并使用反向传播来转换我们的输入图像，以最小化这些损失值。
    为了做到这一点，我们必须加载一个预训练的模型，并使用它学习好的特征映射来描述图像的内容和风格表示/特征。
    我们的main loss function（整体差异损失函数）主要是计算这些不同表示之间的距离
我们通过自定义模型和eager excution实现了这一点
    我们使用Functional API构建了自定义模型
    eager excution允许我们使用自然的的python控制流动态地使用张量
    我们直接操作张量，这使得调试和使用张量更容易。
我们通过使用tf.gradient，应用优化器更新规则rules来不断迭代，更新图像。优化器最小化了给定的与输入图像相关的损失。
Tuebingen图片来源:Andreas Praefcke [GFDL (http://www.gnu.org/copyleft/fdl.html)或CC By 3.0 (https://creativecommons.org/licenses/by/3.0)，来自Wikimedia Commons
绿海龟图像P。Lindgren [CC BY-SA 3.0 (https://creativecommons.org/licenses/bysa/3.0)]，来自Wikimedia Commons


```python

```
