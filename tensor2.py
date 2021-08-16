import tensorflow as tf

import timeit

def method1():
    # The quick method
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    x = tf.Variable(3.0)
    f_x = lambda: x ** 2
    _ = opt.minimize(f_x, var_list=[x])
    return x
    

def method2():
    # The spelled out method
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    x = tf.Variable(3.0)
    f_x = lambda: x ** 2
    with tf.GradientTape() as tape:
        loss = f_x()
    variables = [x]
    grads = tape.gradient(loss, variables)
    opt.apply_gradients(zip(grads, variables))
    return x
    
def method3():
    # Using our own gradients
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    x = tf.Variable([2.0, 3.0, 4.0])
    grads = [tf.tensordot(x, x, 1)] #The gradient at x=3
    opt.apply_gradients(zip(grads, [x]))
    return x
    
x1 = method1()
x2 = method2()
x3 = method3()

assert x1.numpy() == x2.numpy()
assert x2.numpy() == x3.numpy()


t1 = timeit.timeit("method1()", globals=locals(), number=1000)
t2 = timeit.timeit("method2()", globals=locals(), number=1000)
t3 = timeit.timeit("method3()", globals=locals(), number=1000)

