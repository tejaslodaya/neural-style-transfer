import scipy.misc
import tensorflow as tf

from nst_utils import load_vgg_model, reshape_and_normalize_image, generate_noise_image, save_image
from nst_app_utils import compute_content_cost, compute_style_cost, total_cost
from config import CONFIG

# Step 1: Create an interactive session
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Step 2: Load the content image
content_image = scipy.misc.imread(CONFIG.CONTENT_IMAGE)
content_image = reshape_and_normalize_image(content_image)

# Step 3: Load the style image
style_image = scipy.misc.imread(CONFIG.STYLE_IMAGE)
style_image = reshape_and_normalize_image(style_image)

# Step 4: Randomly initialize the image to be generated
generated_image = generate_noise_image(content_image)

# Step 5: Load the VGG16 model
model = load_vgg_model(CONFIG.VGG_MODEL)

# Step 6: Build the tensorflow graph
# Step 6a: Run the content image through VGG16 model and compute content cost
sess.run(model['input'].assign(content_image))
J_content = compute_content_cost(sess, model, CONFIG.CONTENT_LAYER)

# Step 6b: Run the style image through VGG16 model and compute style cost
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(sess, model, CONFIG.STYLE_LAYERS)

# Step 6c: Compute the total cost
J = total_cost(J_content, J_style, alpha = CONFIG.ALPHA, beta = CONFIG.BETA)

# Step 6d: Define the optimizer and learning rate
optimizer = tf.train.AdamOptimizer(CONFIG.LEARNING_RATE)
train_step = optimizer.minimize(J)

# Step 7: Run graph for a large number of iterations, updating the generated image at every step
# Initialize global variable
sess.run(tf.global_variables_initializer())

# Run the noisy initial generated image through the model.
sess.run(model['input'].assign(generated_image))

for i in range(CONFIG.NUM_ITERATIONS):

    # Run the session on the train_step to minimize the total cost
    sess.run(train_step)

    # Compute the generated image by running the session on the current model['input']
    generated_image = sess.run(model['input'])

    # Print every 20 iteration.
    if i % 20 == 0:
        Jt, Jc, Js = sess.run([J, J_content, J_style])
        print("Iteration " + str(i) + " :")
        print("total cost = " + str(Jt))
        print("content cost = " + str(Jc))
        print("style cost = " + str(Js))

        # save current generated image in the "/output" directory
        save_image("output/" + str(i) + ".png", generated_image)

# save last generated image
save_image('output/generated_image.jpg', generated_image)