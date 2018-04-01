# import tensorflow as tf

# x = tf.Variable(3, name = "x")
# y = tf.Variable(4, name = "y")

# f = x * x * y + y + 2

# with tf.Session() as sess:
#     # implicitly call get_default_session()
#     x.initializer.run()  
#     y.initializer.run()

#     # equal to get_default_session().run(f)
#     result = f.eval()


# # Managing graph
# graph = tf.Graph()

# with graph.as_default():
#     x2 = tf.Variable(2)

# x2.graph is graph # -> True
# x2.graph is tf.get_default_graph() # -> False

# tf.reset_default_graph() -> reset all node into default_graph

# NB:
# A variable starts its life when its initializer is 
# run, and it ends when the session is closed.

# w = tf.constant(3)
# x = w + 2
# y = x + 5
# z = x * 3
# with tf.Session() as sess:
#     print(y.eval()) # 10
#     print(z.eval()) # 15

# How to share computation?
# with tf.Session() as sess:
#     y_val, z_val = sess.run([y, z])
#     print(y_val)
#     print(z_val)


# Source ops
# Constants and variables, they take no input

# Tensors
# Multidimensional arrays, they are inputs and outputs
