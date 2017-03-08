# A Whirlwind Tour Through TensorLayer

By TianJun(tianjun.cpp@gmail.com)



```python
In [1]: import tensorlayer as tl

In [2]: tl.__version__
Out[2]: '1.3.11'
```



## Layer!Layer!Layer!

Build model using layers!

```python
# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# define the network
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
```

Ah, it reminds me of [Lasagne](https://github.com/Lasagne/Lasagne).
